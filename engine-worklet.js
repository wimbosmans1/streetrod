(function engineWorkletMain() {
  'use strict';
  const NBQ = 6, NSLOT = 48, DL_LEN = 8192, DL_MASK = DL_LEN - 1;
  const RV_COMB = [1116, 1188, 1277, 1356], RV_AP = [556, 441, 579, 464];
  const C_EXHAUST = 520.0;
  const TWO_PI = 2 * Math.PI;
  // delay line ids
  const DL_H0 = 0, DL_P0 = 2, DL_C0 = 4, DL_A0 = 8, DL_A2 = 10, NDL = 12;

  const DEFAULT_SPEC = {
    name: 'V8', displacement_ci: 289.0, compression: 10.5, idle_rpm: 800.0, redline: 6500.0,
    firing_order: [1, 5, 4, 2, 6, 3, 7, 8], left_bank: [5, 6, 7, 8],
    cam_duration_050: 220.0, lsa: 114.0, lifters: 'solid', carb_cfm: 715.0, air_cleaner: 'open',
    headers: 'tubular', header_len_m: 0.85, pipe_len_m: 3.2, exhaust: 'sport', richness: 1.0,
    cyl_imbalance: 0.06, rev_inertia: 1.0, hcross: 0.15, rev_limiter: false, seed: 1,
  };

  function lopeOf(sp) {
    const l = (sp.cam_duration_050 - 190.0) / 55.0 + (114.0 - sp.lsa) / 16.0;
    return Math.min(1.6, Math.max(0.0, l));
  }

  // ---- RBJ biquad design (identical to _bq) -------------------------------
  function bq(kind, f0, q, fs, gain_db) {
    gain_db = gain_db || 0;
    const A = Math.pow(10, gain_db / 40.0);
    const w0 = 2 * Math.PI * Math.min(f0, 0.45 * fs) / fs;
    const c = Math.cos(w0), s = Math.sin(w0), al = s / (2 * q);
    let b, a;
    if (kind === 'lp') { b = [(1 - c) / 2, 1 - c, (1 - c) / 2]; a = [1 + al, -2 * c, 1 - al]; }
    else if (kind === 'hp') { b = [(1 + c) / 2, -(1 + c), (1 + c) / 2]; a = [1 + al, -2 * c, 1 - al]; }
    else if (kind === 'bp') { b = [al, 0, -al]; a = [1 + al, -2 * c, 1 - al]; }
    else if (kind === 'peak') { b = [1 + al * A, -2 * c, 1 - al * A]; a = [1 + al / A, -2 * c, 1 - al / A]; }
    else if (kind === 'lowshelf') {
      const sq = 2 * Math.sqrt(A) * al;
      b = [A * ((A + 1) - (A - 1) * c + sq), 2 * A * ((A - 1) - (A + 1) * c), A * ((A + 1) - (A - 1) * c - sq)];
      a = [(A + 1) + (A - 1) * c + sq, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - sq];
    } else if (kind === 'highshelf') {
      const sq = 2 * Math.sqrt(A) * al;
      b = [A * ((A + 1) + (A - 1) * c + sq), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - sq)];
      a = [(A + 1) - (A - 1) * c + sq, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - sq];
    } else throw new Error('biquad kind ' + kind);
    return [b[0] / a[0], b[1] / a[0], b[2] / a[0], a[1] / a[0], a[2] / a[0]];
  }

  // ---- numpy.random.RandomState(seed) subset: randn(), rand() ------------
  class NpRandomState {
    constructor(seed) {
      this.mt = new Uint32Array(624); this.idx = 624; this.hasGauss = false; this.gauss = 0;
      const mt = this.mt; mt[0] = seed >>> 0;
      for (let i = 1; i < 624; i++) {
        const p = mt[i - 1] ^ (mt[i - 1] >>> 30);
        mt[i] = (Math.imul(1812433253, p) + i) >>> 0;
      }
    }
    int32() {
      const mt = this.mt;
      if (this.idx >= 624) {
        for (let k = 0; k < 624; k++) {
          const y = (mt[k] & 0x80000000) | (mt[(k + 1) % 624] & 0x7fffffff);
          mt[k] = mt[(k + 397) % 624] ^ (y >>> 1) ^ ((y & 1) ? 0x9908b0df : 0);
        }
        this.idx = 0;
      }
      let y = mt[this.idx++];
      y ^= y >>> 11; y ^= (y << 7) & 0x9d2c5680; y ^= (y << 15) & 0xefc60000; y ^= y >>> 18;
      return y >>> 0;
    }
    double() { const a = this.int32() >>> 5, b = this.int32() >>> 6; return (a * 67108864.0 + b) / 9007199254740992.0; }
    randn() {
      if (this.hasGauss) { this.hasGauss = false; return this.gauss; }
      let x1, x2, r2;
      do { x1 = 2 * this.double() - 1; x2 = 2 * this.double() - 1; r2 = x1 * x1 + x2 * x2; } while (r2 >= 1 || r2 === 0);
      const f = Math.sqrt(-2 * Math.log(r2) / r2);
      this.gauss = f * x1; this.hasGauss = true; return f * x2;
    }
  }

  // ---- the synth ----------------------------------------------------------
  class EngineSynth {
    constructor(specIn, fs) {
      this.spec = Object.assign({}, DEFAULT_SPEC, specIn);
      this.fs = fs || 44100;
      this._build();
      this._rpm = this.spec.idle_rpm; this._thr = 0.0;
    }
    _build() {
      const sp = this.spec, fs = this.fs, lope = lopeOf(sp), dpc = sp.displacement_ci / 8.0;
      const pick = (o, k) => { if (!(k in o)) throw new Error('unknown option ' + k); return o[k]; };
      const P = {};
      P.fs = fs; P.idle = sp.idle_rpm; P.red = sp.redline; P.lope = lope;
      P.sig0 = 0.05; P.sigl = 0.06 + 0.24 * lope; P.misf = 0.005 + 0.07 * lope * lope;
      P.huntd = 0.004 + 0.022 * lope; P.hunthz = 0.7 + 0.6 * lope; P.kick = 0.008 + 0.05 * lope;
      P.lopespan = 900.0 + 1300.0 * lope; P.idlemap = 0.30 + 0.10 * lope; P.ovmap = 0.07;
      P.vepk = 2600.0 + 2000.0 * (sp.cam_duration_050 - 190.0) / 60.0; P.vew = 2200.0;
      P.blowdeg = 52.0 + 18.0 * lope; P.wmin = 0.45e-3 * fs; P.wmax = (1.6 + 0.035 * dpc) * 1e-3 * fs;
      P.att = Math.min(0.35, Math.max(0.12, 0.26 - 0.015 * (sp.compression - 8.0)));
      P.amp = (dpc / 36.0) * (0.85 + 0.05 * sp.compression / 10.0);
      P.rasp = pick({ turbo: 0.10, sport: 0.22, glasspack: 0.30, open: 0.42 }, sp.exhaust) + 0.06 * lope;
      P.jit = 0.12 + 0.25 * lope;
      P.popr = pick({ turbo: 0.006, sport: 0.035, glasspack: 0.06, open: 0.08 }, sp.exhaust) * (0.5 + lope);
      P.popa = 1.5;
      P.tick = sp.lifters === 'solid' ? 1.2 : 0.12;
      P.intg = 0.35 * Math.sqrt(sp.carb_cfm / 600.0) * (sp.air_cleaner === 'open' ? 1.0 : 0.6);
      P.intrev = 0.3 * lope;
      P.mech = sp.lifters === 'solid' ? 0.0035 : 0.002;
      if (sp.headers === 'tubular') { P.hfb = -0.55; P.hdamp = 0.45; } else { P.hfb = -0.30; P.hdamp = 0.25; }
      P.hdly = Math.max(8, pyRound(2 * sp.header_len_m / C_EXHAUST * fs));
      if (sp.exhaust === 'open') { P.pdly = 16; P.pfb = 0.0; P.pdamp = 0.5; }
      else {
        P.pdly = pyRound(2 * sp.pipe_len_m / C_EXHAUST * fs);
        P.pfb = pick({ turbo: -0.35, sport: -0.45, glasspack: -0.5 }, sp.exhaust);
        P.pdamp = pick({ turbo: 0.18, sport: 0.3, glasspack: 0.35 }, sp.exhaust);
      }
      const rad_fc = pick({ turbo: 45.0, sport: 55.0, glasspack: 90.0, open: 130.0 }, sp.exhaust);
      P.radc = 1.0 / (1.0 + 2 * Math.PI * rad_fc / fs);
      P.panw = 0.78; P.hx = sp.hcross; P.revmix = 0.035; P.cfm = sp.carb_cfm; P.disp = sp.displacement_ci;
      P.limit = sp.rev_limiter ? 1.0 : 0.0;
      P.thrk = 1.0 - Math.exp(-1.0 / (0.045 * fs));
      P.rich = sp.richness;
      P.jet = pick({ turbo: 0.5, sport: 2.2, glasspack: 2.0, open: 4.0 }, sp.exhaust) * (1.0 + 0.3 * lope);
      P.jetk = 1.0 - Math.exp(-1.0 / (0.0006 * fs));
      P.exgain = 1.0;
      this.P = P;

      // firing order -> bank, bias, primary delay (numpy RandomState(seed))
      const rs = new NpRandomState(sp.seed);
      const fo = sp.firing_order.slice();
      this.ev_bank = new Int32Array(8);
      for (let k = 0; k < 8; k++) this.ev_bank[k] = sp.left_bank.indexOf(fo[k]) >= 0 ? 0 : 1;
      const bias = []; for (let k = 0; k < 8; k++) bias.push(1.0 + sp.cyl_imbalance * rs.randn());
      // numpy's pairwise sum for exactly 8 elements: ((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7))
      const m = (((bias[0] + bias[1]) + (bias[2] + bias[3])) + ((bias[4] + bias[5]) + (bias[6] + bias[7]))) / 8;
      this.ev_bias = new Float64Array(bias.map((b) => b / m));
      this.ev_dly = new Float64Array(8);
      if (sp.headers === 'tubular') {
        for (let k = 0; k < 8; k++) this.ev_dly[k] = 0.06e-3 * fs * rs.double();
      } else {
        const lb = sp.left_bank.slice().sort().join(',');
        for (let k = 0; k < 8; k++) {
          const c = fo[k];
          const pos = lb === '5,6,7,8' ? ((c - 1) % 4) : Math.floor((c - 1) / 2);
          this.ev_dly[k] = (0.15 + 0.22 * pos) / C_EXHAUST * fs;
        }
      }
      const EX = {
        turbo: [['lp', 300, 0.75, 0], ['lp', 650, 0.6, 0], ['peak', 75, 1.1, 5.0], ['peak', 160, 1.4, 2.0], ['highshelf', 1500, 0.7, -8]],
        sport: [['lp', 2000, 0.8, 0], ['lowshelf', 110, 0.7, 5.0], ['peak', 420, 1.2, 3.0], ['peak', 1600, 2.0, 2.0], ['lp', 4500, 0.6, 0]],
        glasspack: [['lp', 2200, 0.7, 0], ['peak', 170, 1.0, 4.0], ['peak', 800, 1.3, 4.0], ['peak', 2600, 2.0, 2.5], ['lp', 6500, 0.6, 0]],
        open: [['lp', 7000, 0.6, 0], ['peak', 650, 1.0, 3.0], ['peak', 2200, 1.5, 3.0], ['highshelf', 4000, 0.7, -3]],
      };
      const ex = pick(EX, sp.exhaust);
      this.ex_n = ex.length;
      this.ex_co = new Float64Array(2 * NBQ * 5); this.ex_st = new Float64Array(2 * NBQ * 2);
      for (let b = 0; b < 2; b++) ex.forEach(([k, f, q, g], j) => {
        this.ex_co.set(bq(k, f * (1.0 + 0.04 * (b - 0.5)), q, fs, g), (b * NBQ + j) * 5);
      });
      const hz = 230.0 * Math.sqrt(sp.carb_cfm / 600.0);
      const inn = [['bp', hz, 1.6, 0], ['peak', 2 * hz, 2.0, 3.0], ['lp', sp.air_cleaner === 'open' ? 3500 : 1400, 0.7, 0], ['hp', 60, 0.7, 0]];
      this.in_n = inn.length; this.in_co = new Float64Array(NBQ * 5); this.in_st = new Float64Array(NBQ * 2);
      inn.forEach(([k, f, q, g], j) => this.in_co.set(bq(k, f, q, fs, g), j * 5));
      const mk = [['bp', 3300, 9.0, 0], ['bp', 5400, 12.0, 0], ['bp', 1600, 0.8, 0]];
      this.mk_co = new Float64Array(3 * 5); this.mk_st = new Float64Array(3 * 2);
      mk.forEach(([k, f, q, g], j) => this.mk_co.set(bq(k, f, q, fs, g), j * 5));
      const JET = {
        turbo: [['hp', 500, 0.7, 0], ['lp', 1800, 0.7, 0], ['peak', 900, 1.0, 0]],
        sport: [['hp', 700, 0.7, 0], ['lp', 4200, 0.7, 0], ['peak', 1800, 1.2, 3.0]],
        glasspack: [['hp', 600, 0.7, 0], ['lp', 5000, 0.7, 0], ['peak', 1500, 1.2, 4.0]],
        open: [['hp', 450, 0.7, 0], ['lp', 7500, 0.7, 0], ['peak', 2500, 1.0, 3.0]],
      };
      const jet = pick(JET, sp.exhaust);
      this.jt_n = jet.length; this.jt_co = new Float64Array(2 * 3 * 5); this.jt_st = new Float64Array(2 * 3 * 2);
      for (let b = 0; b < 2; b++) jet.forEach(([k, f, q, g], j) => this.jt_co.set(bq(k, f * (1 + 0.05 * b), q, fs, g), (b * 3 + j) * 5));

      // state
      this.S = {
        th: 0, thr: 0, lope: 0, hunt: 0, limtog: 0, unb: 0, rad0: 0, rad1: 0, rady0: 0, rady1: 0,
        hlp0: 0, hlp1: 0, plp0: 0, plp1: 0, nz0: 0, nz1: 0, nz2: 0, rv0: 0, rv1: 0, rv2: 0, rv3: 0, ovf: 0, je0: 0, je1: 0,
      };
      // xorshift64 state as two uint32 halves: 0x9E3779B97F4A7C15 ^ (seed*7919)
      const sx = sp.seed * 7919;
      this.rhi = (0x9E3779B9 ^ Math.floor(sx / 4294967296)) >>> 0;
      this.rlo = (0x7F4A7C15 ^ (sx >>> 0)) >>> 0;
      this.slots = new Float64Array(NSLOT * 6);
      for (let i = 0; i < NSLOT; i++) this.slots[i * 6] = -1;
      this.dl = new Float64Array(NDL * DL_LEN); this.di = new Int32Array(NDL);
      const sc = fs / 44100.0;
      this.rv_d = RV_COMB.concat(RV_AP).map((d) => Math.trunc(d * sc));
    }

    _rand() {
      let hi = this.rhi, lo = this.rlo, h, l;
      h = hi ^ ((hi << 13) | (lo >>> 19)); l = lo ^ (lo << 13); hi = h >>> 0; lo = l >>> 0;
      l = lo ^ ((lo >>> 7) | (hi << 25)); h = hi ^ (hi >>> 7); hi = h >>> 0; lo = l >>> 0;
      h = hi ^ ((hi << 17) | (lo >>> 15)); l = lo ^ (lo << 17); hi = h >>> 0; lo = l >>> 0;
      this.rhi = hi; this.rlo = lo;
      return ((hi >>> 11) * 4294967296 + (((hi << 21) | (lo >>> 11)) >>> 0)) * (1.0 / 9007199254740992.0);
    }
    _gauss() { return (this._rand() + this._rand() + this._rand() + this._rand() - 2.0) * 1.732; }

    _spawn(kind, start, width, att, amp, nz) {
      const s = this.slots; let best = 0, bestpos = -1e30;
      for (let i = 0; i < NSLOT; i++) {
        const o = i * 6;
        if (s[o] < 0) { best = i; break; }
        if (s[o + 1] > bestpos) { bestpos = s[o + 1]; best = i; }
      }
      const o = best * 6;
      s[o] = kind; s[o + 1] = start; s[o + 2] = width; s[o + 3] = att; s[o + 4] = amp; s[o + 5] = nz;
    }

    /** render n frames into outL/outR (offset off); rpm/throttle ramp linearly
     *  from the previous call's targets to these (exactly like render()). */
    render(n, rpm, throttle, outL, outR, off, stemE, stemF) {
      off = off || 0;
      const r0 = this._rpm, t0 = this._thr;
      this._rpm = rpm; this._thr = throttle;
      const P = this.P, S = this.S, slots = this.slots, dl = this.dl, di = this.di, rv_d = this.rv_d;
      const ex_co = this.ex_co, ex_st = this.ex_st, in_co = this.in_co, in_st = this.in_st;
      const mk_co = this.mk_co, mk_st = this.mk_st, jt_co = this.jt_co, jt_st = this.jt_st;
      const fs = P.fs, idle = P.idle, red = P.red;
      const hunt_inc = 2.0 * Math.PI * P.hunthz / fs;
      const hdly = Math.trunc(P.hdly), pdly = Math.trunc(P.pdly);
      for (let n_ = 0; n_ < n; n_++) {
        const tt = (n_ + 1) / n;
        const rpm_t = r0 + (rpm - r0) * tt;
        const thr_in = t0 + (throttle - t0) * tt;
        const thr = S.thr + (thr_in - S.thr) * P.thrk;
        S.thr = thr;
        let idle_w = 1.0 - (rpm_t - idle) / P.lopespan;
        if (idle_w < 0.0) idle_w = 0.0; else if (idle_w > 1.0) idle_w = 1.0;
        S.hunt += hunt_inc;
        if (S.hunt > TWO_PI) S.hunt -= TWO_PI;
        const hunt = P.huntd * idle_w * (1.0 - thr) * Math.sin(S.hunt);
        const rpmv = rpm_t * (1.0 + S.lope + hunt);
        S.lope *= 0.99985;
        const dth = rpmv * 6.0 / fs;
        let ov = 0.0;
        if (thr < 0.06) {
          ov = (rpmv - idle * 1.35) / (idle * 0.9);
          if (ov < 0.0) ov = 0.0; else if (ov > 1.0) ov = 1.0;
          ov *= (0.06 - thr) / 0.06;
        }
        S.ovf = ov;
        const th0 = S.th;
        let th1 = th0 + dth;
        const k0 = Math.floor(th0 / 90.0), k1 = Math.floor(th1 / 90.0);
        if (k1 !== k0) {
          const k = ((k1 % 8) + 8) % 8;
          const since = (th1 - k1 * 90.0) / dth;
          const bank = this.ev_bank[k];
          const flow = P.disp * rpmv / 3456.0 * 0.9;
          let restr = 1.0;
          if (flow > 1.0) { restr = P.cfm * 1.15 / flow; if (restr > 1.0) restr = 1.0; }
          const mp_open = P.idlemap + (1.0 - P.idlemap) * thr * (0.75 + 0.25 * restr);
          const load = mp_open * (1.0 - ov) + P.ovmap * ov;
          const x = (rpmv - P.vepk) / P.vew;
          const ve = 0.75 + 0.25 * Math.exp(-x * x);
          const sig = P.sig0 + P.sigl * idle_w * (1.0 - thr) + 0.5 * ov;
          let a = load * ve * this.ev_bias[k] * (1.0 + sig * this._gauss());
          if (a < 0.02 * load) a = 0.02 * load;
          if (this._rand() < P.misf * idle_w * (1.0 - thr) + 0.08 * ov) {
            a *= 0.15 + 0.3 * this._rand();
            S.unb += 1.0;
          }
          if (P.limit > 0.5 && rpm_t >= red) {
            S.limtog = 1.0 - S.limtog;
            if (S.limtog > 0.5) { a = 0.0; S.unb += 1.0; }
          }
          S.lope += P.kick * idle_w * (1.0 - thr) * (a / (load * ve * this.ev_bias[k] + 1e-6) - 1.0) * 0.25;
          if (S.lope > 0.08) S.lope = 0.08; else if (S.lope < -0.08) S.lope = -0.08;
          let w = P.blowdeg / dth;
          if (w > P.wmax) w = P.wmax;
          if (w < P.wmin) w = P.wmin;
          let att = P.att * (1.3 - 0.85 * load);
          if (att * w < 5.0) att = 5.0 / w;
          const jit = P.jit * (0.3 + idle_w) * this._gauss();
          const start = since - this.ev_dly[k] - jit * fs * 1e-3;
          const nz = P.rasp * (0.35 + 0.65 * load);
          this._spawn(bank, start, w, att, a * P.amp, nz);
          let wi = 180.0 / dth;
          if (wi > 0.02 * fs) wi = 0.02 * fs;
          const ia = P.intg * (thr * (0.3 + 0.7 * rpmv / red) + P.intrev * idle_w * (1.0 - thr) * (0.6 + 0.8 * this._rand()) + 0.03);
          this._spawn(2, since - 0.25 * wi, wi, 0.35, ia, 1.0);
          const ta = P.tick * (0.4 + 0.6 * rpmv / red) * (0.7 + 0.6 * this._rand());
          this._spawn(3, since - 0.3 * 90.0 / dth, 1.0, 0.0, ta, 0.0);
          this._spawn(3, since - 0.7 * 90.0 / dth, 1.0, 0.0, ta * (0.6 + 0.5 * this._rand()), 0.0);
          const pr = P.popr * P.rich * (ov * (0.4 + 0.6 * Math.min(1.0, (rpmv - idle) / 2500.0)) + 0.4 * Math.min(S.unb, 3.0) / 3.0 * ov);
          if (this._rand() < pr) {
            const pw = (0.6 + 0.8 * this._rand()) * 1e-3 * fs;
            const pa = P.popa * P.amp * (0.4 + 0.9 * this._rand());
            this._spawn(4 + bank, since - (1.0 + 25.0 * this._rand()) * 1e-3 * fs, pw, 0.12, pa, 1.8);
            S.unb = 0.0;
          }
          S.unb *= 0.9;
        }
        th1 = th1 % 720.0;   // th1 >= 0 always, so JS % == Python %
        S.th = th1;

        // ---- active pulses ----
        let exh0 = 0, exh1 = 0, pop0 = 0, pop1 = 0, intk = 0, tick = 0, je0 = 0, je1 = 0;
        S.nz0 += 0.55 * ((this._rand() * 2.0 - 1.0) - S.nz0);
        S.nz1 += 0.55 * ((this._rand() * 2.0 - 1.0) - S.nz1);
        S.nz2 += 0.35 * ((this._rand() * 2.0 - 1.0) - S.nz2);
        for (let i = 0; i < NSLOT; i++) {
          const o = i * 6;
          const kind = slots[o];
          if (kind < 0) continue;
          const pos = slots[o + 1];
          slots[o + 1] = pos + 1.0;
          if (pos < 0.0) continue;
          const ik = kind | 0;
          if (ik === 3) { tick += slots[o + 4]; slots[o] = -1.0; continue; }
          const xw = pos / slots[o + 2];
          if (xw >= 1.0) { slots[o] = -1.0; continue; }
          const at = slots[o + 3];
          let env;
          if (xw < at) env = 0.5 - 0.5 * Math.cos(Math.PI * xw / at);
          else { const u = (xw - at) / (1.0 - at); env = 0.5 + 0.5 * Math.cos(Math.PI * u); env *= env; }
          const amp = slots[o + 4] * env, nzv = slots[o + 5];
          if (ik === 0) { je0 += amp; exh0 += amp * (1.0 + nzv * S.nz0 * 3.0); }
          else if (ik === 1) { je1 += amp; exh1 += amp * (1.0 + nzv * S.nz1 * 3.0); }
          else if (ik === 2) intk += amp * (0.25 + nzv * S.nz2 * 2.5);
          else if (ik === 4) pop0 += amp * (1.0 + nzv * S.nz0 * 2.0);
          else pop1 += amp * (1.0 + nzv * S.nz1 * 2.0);
        }
        const hx = P.hx;
        // header combs
        let ys0 = 0, ys1 = 0;
        for (let b = 0; b < 2; b++) {
          const xin = b === 0 ? exh0 : exh1;
          const hl = DL_H0 + b, base = hl * DL_LEN, idx = di[hl];
          const fbv = dl[base + ((idx - hdly) & DL_MASK)];
          const key = b === 0 ? 'hlp0' : 'hlp1';
          S[key] += P.hdamp * (fbv - S[key]);
          const yh = xin + P.hfb * S[key];
          dl[base + idx] = yh; di[hl] = (idx + 1) & DL_MASK;
          if (b === 0) ys0 = yh; else ys1 = yh;
        }
        for (let b = 0; b < 2; b++) {
          // NB: as in v1 Python, ys0 is already overwritten by bank 0's tailpipe
          // output when bank 1 reads its crossover input. Kept for parity.
          const yh = b === 0 ? ys0 : ys1, yo = b === 0 ? ys1 : ys0;
          const xp = (1.0 - hx) * yh + hx * yo + (b === 0 ? pop0 : pop1);
          const pl = DL_P0 + b, base = pl * DL_LEN, idx = di[pl];
          const fbv = dl[base + ((idx - pdly) & DL_MASK)];
          const key = b === 0 ? 'plp0' : 'plp1';
          S[key] += P.pdamp * (fbv - S[key]);
          const yp = xp + P.pfb * S[key];
          dl[base + idx] = yp; di[pl] = (idx + 1) & DL_MASK;
          let v = yp;
          for (let j = 0; j < this.ex_n; j++) {
            const c = (b * NBQ + j) * 5, s = (b * NBQ + j) * 2;
            const y = ex_co[c] * v + ex_st[s];
            ex_st[s] = ex_co[c + 1] * v - ex_co[c + 3] * y + ex_st[s + 1];
            ex_st[s + 1] = ex_co[c + 2] * v - ex_co[c + 4] * y;
            v = y;
          }
          const r = P.radc;
          const rk = b === 0 ? 'rad0' : 'rad1', ryk = b === 0 ? 'rady0' : 'rady1';
          let yr = r * (S[ryk] + v - S[rk]);
          S[rk] = v; S[ryk] = yr;
          const je = b === 0 ? je0 : je1, jk = b === 0 ? 'je0' : 'je1';
          S[jk] += P.jetk * (Math.abs(je) - S[jk]);
          const ev = S[jk] / (P.amp + 1e-9);
          let jn = (this._rand() * 2.0 - 1.0) * ev * ev * P.amp * P.jet * (0.45 + rpmv / red);
          for (let j = 0; j < 3; j++) {
            const c = (b * 3 + j) * 5, s = (b * 3 + j) * 2;
            const y = jt_co[c] * jn + jt_st[s];
            jt_st[s] = jt_co[c + 1] * jn - jt_co[c + 3] * y + jt_st[s + 1];
            jt_st[s + 1] = jt_co[c + 2] * jn - jt_co[c + 4] * y;
            jn = y;
          }
          yr += jn;
          if (b === 0) ys0 = yr; else ys1 = yr;
        }
        const ex0 = ys0 * P.exgain, ex1 = ys1 * P.exgain;
        let vi = intk;
        for (let j = 0; j < this.in_n; j++) {
          const c = j * 5, s = j * 2;
          const y = in_co[c] * vi + in_st[s];
          in_st[s] = in_co[c + 1] * vi - in_co[c + 3] * y + in_st[s + 1];
          in_st[s + 1] = in_co[c + 2] * vi - in_co[c + 4] * y;
          vi = y;
        }
        // mechanical
        let vm = 0;
        for (let j = 0; j < 2; j++) {
          const c = j * 5, s = j * 2;
          const y = mk_co[c] * tick + mk_st[s];
          mk_st[s] = mk_co[c + 1] * tick - mk_co[c + 3] * y + mk_st[s + 1];
          mk_st[s + 1] = mk_co[c + 2] * tick - mk_co[c + 4] * y;
          vm += j === 0 ? y : 0.6 * y;
        }
        let mn;
        {
          const xin = this._rand() * 2.0 - 1.0;
          const c = 10, s = 4;
          const y = mk_co[c] * xin + mk_st[s];
          mk_st[s] = mk_co[c + 1] * xin - mk_co[c + 3] * y + mk_st[s + 1];
          mk_st[s + 1] = mk_co[c + 2] * xin - mk_co[c + 4] * y;
          mn = y * P.mech * (0.2 + rpmv / red);
        }
        const mech = vm + mn;
        if (stemE) { stemE[off + n_] = ex0 + ex1; stemF[off + n_] = 0.8 * vi + 0.7 * mech; }   // optional stems (cabin model)
        const pw = P.panw;
        const L = pw * ex1 + (1.0 - pw) * ex0 + 0.8 * vi + 0.7 * mech;
        const R = pw * ex0 + (1.0 - pw) * ex1 + 0.8 * vi + 0.6 * mech;
        // ambience
        const mm = 0.5 * (L + R);
        let acc = 0.0;
        for (let c = 0; c < 4; c++) {
          const line = DL_C0 + c, base = line * DL_LEN, idx = di[line];
          const yv = dl[base + ((idx - rv_d[c]) & DL_MASK)];
          const key = 'rv' + c;
          S[key] += 0.6 * (yv - S[key]);
          dl[base + idx] = mm + 0.62 * S[key]; di[line] = (idx + 1) & DL_MASK;
          acc += yv;
        }
        let wl = acc, wr = acc;
        for (let c = 0; c < 2; c++) {
          let line = DL_A0 + c, base = line * DL_LEN, idx = di[line];
          let bv = dl[base + ((idx - rv_d[4 + c]) & DL_MASK)];
          dl[base + idx] = wl + 0.5 * bv; di[line] = (idx + 1) & DL_MASK;
          wl = bv - 0.5 * wl;
          line = DL_A2 + c; base = line * DL_LEN; idx = di[line];
          bv = dl[base + ((idx - rv_d[6 + c]) & DL_MASK)];
          dl[base + idx] = wr + 0.5 * bv; di[line] = (idx + 1) & DL_MASK;
          wr = bv - 0.5 * wr;
        }
        const rm = P.revmix;
        outL[off + n_] = L + rm * wl;
        outR[off + n_] = R + rm * wr;
      }
    }
  }

  function pyRound(x) {   // Python round(): banker's rounding on exact .5
    const f = Math.floor(x), d = x - f;
    if (d > 0.5) return f + 1; if (d < 0.5) return f;
    return (f % 2 === 0) ? f : f + 1;
  }

  // ---- extras for the game (not part of the v1 model) ---------------------
  /** starter motor + cranking compression pulses */
  class Starter {
    constructor(fs) { this.fs = fs; this.ph = 0; this.ph2 = 0; this.lp = 0; this.bp = [0, 0]; this.seed = 12345; }
    rnd() { this.seed = (Math.imul(this.seed, 1664525) + 1013904223) >>> 0; return this.seed / 4294967296 * 2 - 1; }
    sample(level, crankRpm) {
      const fs = this.fs;
      this.ph += 2 * Math.PI * (crankRpm / 60 * 4) / fs;        // 4 compressions per rev (V8)
      if (this.ph > 2 * Math.PI) this.ph -= 2 * Math.PI;
      this.ph2 += 2 * Math.PI * (crankRpm / 60 * 11) / fs;       // pinion whine
      if (this.ph2 > 2 * Math.PI) this.ph2 -= 2 * Math.PI;
      const comp = Math.pow(0.5 + 0.5 * Math.sin(this.ph), 6);
      this.lp += 0.05 * (this.rnd() - this.lp);
      const whine = 0.25 * Math.sin(this.ph2) + 0.12 * Math.sin(2 * this.ph2);
      return level * (0.35 * comp * (0.6 + this.lp) + whine * (0.4 + 0.6 * comp) + 0.08 * this.lp);
    }
  }

  /** tyre noise: road roar (speed) + skid screech (slip intensity) */
  class TyreNoise {
    constructor(fs) {
      this.fs = fs; this.seed = 987654; this.r1 = [0, 0]; this.r2 = [0, 0]; this.s1 = [0, 0]; this.s2 = [0, 0];
      this.road = bq('lp', 350, 0.7, fs); this.road2 = bq('peak', 120, 1.0, fs, 6);
      this.sq1 = bq('bp', 900, 6, fs); this.sq2 = bq('bp', 1900, 4, fs); this.ph = 0; this.mod = 0;
    }
    rnd() { this.seed = (Math.imul(this.seed, 1664525) + 1013904223) >>> 0; return this.seed / 4294967296 * 2 - 1; }
    static bqs(x, co, st) { const y = co[0] * x + st[0]; st[0] = co[1] * x - co[3] * y + st[1]; st[1] = co[2] * x - co[4] * y; return y; }
    sample(roadLvl, skidLvl) {
      const n = this.rnd();
      let r = TyreNoise.bqs(n, this.road, this.r1); r = TyreNoise.bqs(r, this.road2, this.r2);
      this.mod += 0.0004 * (this.rnd() - this.mod * 0.02);
      this.ph += 2 * Math.PI * (820 + 140 * Math.sin(this.mod * 50)) / this.fs;
      if (this.ph > 2 * Math.PI) this.ph -= 2 * Math.PI;
      const sq = TyreNoise.bqs(n, this.sq1, this.s1) * 2 + TyreNoise.bqs(n, this.sq2, this.s2) * 1.2 + 0.35 * Math.sin(this.ph) * (0.7 + 0.3 * n);
      return roadLvl * r * 0.6 + skidLvl * sq;
    }
  }

  /** look-ahead peak limiter (ceiling dBFS), stereo-linked; O(1) sliding-window min */
  class Limiter {
    constructor(fs, ceilingDb, lookMs, releaseMs) {
      this.ceil = Math.pow(10, (ceilingDb === undefined ? -1 : ceilingDb) / 20);
      this.n = Math.max(1, Math.round((lookMs || 3) * 1e-3 * fs));
      this.bufL = new Float32Array(this.n); this.bufR = new Float32Array(this.n);
      this.i = 0; this.g = 1; this.rel = 1 - Math.exp(-1 / ((releaseMs || 180) * 1e-3 * fs));
      this.att = 1 - Math.exp(-1 / (0.6 * this.n)); this.minG = 1;
      // monotonic deque of (value, time) for the window minimum
      this.qv = new Float32Array(this.n + 1); this.qt = new Float64Array(this.n + 1);
      this.qh = 0; this.qn = 0; this.t = 0;
    }
    process(L, R, oL, oR, n) {
      const N = this.n, cap = N + 1, qv = this.qv, qt = this.qt;
      for (let k = 0; k < n; k++) {
        const x = Math.max(Math.abs(L[k]), Math.abs(R[k]));
        const need = x > this.ceil ? this.ceil / x : 1;
        const t = this.t++;
        // push back, popping larger values
        while (this.qn > 0 && qv[(this.qh + this.qn - 1) % cap] >= need) this.qn--;
        const tail = (this.qh + this.qn) % cap; qv[tail] = need; qt[tail] = t; this.qn++;
        while (qt[this.qh] <= t - N) { this.qh = (this.qh + 1) % cap; this.qn--; }
        const target = qv[this.qh];
        const i = this.i, dL = this.bufL[i], dR = this.bufR[i];
        this.bufL[i] = L[k]; this.bufR[i] = R[k]; this.i = (i + 1) % N;
        if (target < this.g) this.g += (target - this.g) * this.att; else this.g += (target - this.g) * this.rel;
        let yl = dL * this.g, yr = dR * this.g;
        if (yl > this.ceil || yl < -this.ceil) yl = Math.tanh(yl / this.ceil) * this.ceil;
        if (yr > this.ceil || yr < -this.ceil) yr = Math.tanh(yr / this.ceil) * this.ceil;
        oL[k] = yl; oR[k] = yr;
        if (this.g < this.minG) this.minG = this.g;
      }
    }
  }

  function bqs(x, co, st) { const y = co[0] * x + st[0]; st[0] = co[1] * x - co[3] * y + st[1]; st[1] = co[2] * x - co[4] * y; return y; }

  /**
   * EngineVoice: everything the game hears, independent of the audio API, so it
   * runs inside the AudioWorklet or (fallback) a main-thread ScriptProcessor.
   * process(L, R, n) with n <= 128 per call (call repeatedly for bigger buffers).
   */
  class EngineVoice {
    constructor(fs, o) {
      o = o || {};
      this.fs = fs; this.raw = !!o.raw;
      this.synth = null; this.next = null; this.xfade = 1;
      this.target = { rpm: 800, thr: 0 }; this.sm = { rpm: 800, thr: 0 };
      this.smK = 1 - Math.exp(-128 / (0.025 * fs));
      this.engGain = 0; this.engGainTarget = 0; this.gainK = 1 - Math.exp(-1 / (0.03 * fs));
      this.outGain = o.outGain || 1; this.outGainS = this.outGain;
      this.starter = new Starter(fs); this.starterLvl = 0; this.starterTarget = 0; this.crank = 200;
      this.tyre = new TyreNoise(fs); this.skid = 0; this.skidS = 0; this.speed = 0; this.speedS = 0;
      this.limiter = new Limiter(fs, -1.0, 3, 180);
      this.script = null; this.scriptPos = 0;
      const B = 128;
      this.tL = new Float32Array(B); this.tR = new Float32Array(B); this.tL2 = new Float32Array(B); this.tR2 = new Float32Array(B);
      this.sE = new Float32Array(B); this.sF = new Float32Array(B); this.sE2 = new Float32Array(B); this.sF2 = new Float32Array(B);
      // ---- in-cabin model ----
      // exhaust heard through the body/floor: strong low-pass, cabin boom 40-80 Hz
      this.cabE = [bq('lp', 220, 0.8, fs), bq('lp', 480, 0.7, fs), bq('peak', 58, 2.2, fs, 9), bq('peak', 115, 1.6, fs, 3), bq('hp', 25, 0.7, fs)];
      // intake / engine bay / valvetrain through the firewall: brighter, mid-forward
      this.cabF = [bq('lp', 2600, 0.7, fs), bq('peak', 420, 1.0, fs, 4), bq('peak', 1100, 1.5, fs, 2), bq('hp', 70, 0.7, fs)];
      // small cabin "box" reflections
      this.cabD = new Float32Array(2048); this.cabDi = 0;
      this.cabEst = this.cabE.map(() => [0, 0]); this.cabFst = this.cabF.map(() => [0, 0]);
      // wind noise (speed^3 power) and road roar
      this.seed = 4242; this.wind = [bq('bp', 700, 0.5, fs), bq('lp', 3000, 0.7, fs)]; this.windSt = [[0, 0], [0, 0]]; this.windLfo = 0;
      this.listen = 0; this.listenS = 0; this.listenK = 1 - Math.exp(-1 / (0.35 * fs));
      this.cabGain = o.cabGain || 2.2;
      this.blocks = 0; this.gr = 1;
      if (o.spec) this.synth = new EngineSynth(o.spec, fs);
      if (o.script) { this.script = o.script; this.engGain = this.engGainTarget = 1; }
    }
    rnd() { this.seed = (Math.imul(this.seed, 1664525) + 1013904223) >>> 0; return this.seed / 4294967296 * 2 - 1; }
    msg(m) {
      if (m.type === 'spec') {
        const s = new EngineSynth(m.spec, this.fs);
        if (m.outGain) this.outGain = m.outGain;
        if (this.synth && this.engGain > 0.001) { this.next = s; this.xfade = 0; }
        else { this.synth = s; this.sm.rpm = this.target.rpm = s.spec.idle_rpm; s._rpm = s.spec.idle_rpm; }
      } else if (m.type === 'ctl') {
        if (m.rpm !== undefined) this.target.rpm = m.rpm;
        if (m.thr !== undefined) this.target.thr = m.thr;
        if (m.running !== undefined) this.engGainTarget = m.running ? 1 : 0;
        if (m.starter !== undefined) this.starterTarget = m.starter ? 1 : 0;
        if (m.crank !== undefined) this.crank = m.crank;
        if (m.speed !== undefined) this.speed = m.speed;
        if (m.skid !== undefined) this.skid = m.skid;
        if (m.listen !== undefined) this.listen = m.listen;
      } else if (m.type === 'script') { this.script = m.script; this.scriptPos = 0; }
    }
    process(L, R, n) {
      this.blocks++;
      if (!this.synth) { for (let k = 0; k < n; k++) { L[k] = 0; R[k] = 0; } return; }
      let rpm, thr;
      if (this.script) {
        const i = Math.min(this.scriptPos, this.script.length - 1);
        rpm = this.script[i][0]; thr = this.script[i][1]; this.scriptPos++;
      } else {
        this.sm.rpm += (this.target.rpm - this.sm.rpm) * this.smK;
        this.sm.thr += (this.target.thr - this.sm.thr) * this.smK;
        rpm = Math.max(60, this.sm.rpm); thr = Math.min(1, Math.max(0, this.sm.thr));
      }
      const tL = this.tL, tR = this.tR, sE = this.sE, sF = this.sF;
      this.synth.render(n, rpm, thr, tL, tR, 0, sE, sF);
      if (this.raw) { for (let k = 0; k < n; k++) { L[k] = tL[k]; R[k] = tR[k]; } return; }
      if (this.next) {
        this.next.render(n, rpm, thr, this.tL2, this.tR2, 0, this.sE2, this.sF2);
        for (let k = 0; k < n; k++) {
          this.xfade = Math.min(1, this.xfade + 1 / (0.15 * this.fs));
          const a = Math.cos(this.xfade * Math.PI / 2), b = Math.sin(this.xfade * Math.PI / 2);
          tL[k] = a * tL[k] + b * this.tL2[k]; tR[k] = a * tR[k] + b * this.tR2[k];
          sE[k] = a * sE[k] + b * this.sE2[k]; sF[k] = a * sF[k] + b * this.sF2[k];
        }
        if (this.xfade >= 1) { this.synth = this.next; this.next = null; }
      }
      const fs = this.fs;
      for (let k = 0; k < n; k++) {
        this.engGain += (this.engGainTarget - this.engGain) * this.gainK;
        this.starterLvl += (this.starterTarget - this.starterLvl) * this.gainK;
        this.speedS += (this.speed - this.speedS) * 0.0005; this.skidS += (this.skid - this.skidS) * 0.002;
        this.outGainS += (this.outGain - this.outGainS) * 0.0005;
        this.listenS += (this.listen - this.listenS) * this.listenK;
        const g = this.engGain * this.outGainS, w = this.listenS;
        let eL = tL[k] * g, eR = tR[k] * g;
        const v = this.speedS, vr = v / 40;
        const road = Math.min(0.08, 0.05 * vr * vr);
        // wind: broadband, gusty, rising ~ v^3 in power
        let wn = 0;
        if (v > 3) {
          this.windLfo += 2 * Math.PI * 0.3 / fs; if (this.windLfo > 6.2832) this.windLfo -= 6.2832;
          wn = bqs(bqs(this.rnd(), this.wind[0], this.windSt[0]), this.wind[1], this.windSt[1]);
          wn *= 0.02 * Math.pow(vr, 1.5) * (0.85 + 0.15 * Math.sin(this.windLfo));
        }
        if (w > 1e-3) {
          let ce = sE[k] * g; for (let j = 0; j < this.cabE.length; j++) ce = bqs(ce, this.cabE[j], this.cabEst[j]);
          let cf = sF[k] * g; for (let j = 0; j < this.cabF.length; j++) cf = bqs(cf, this.cabF[j], this.cabFst[j]);
          let c = this.cabGain * (0.9 * ce + 0.55 * cf);
          // short box reflections (~3, 5.5, 8 ms)
          const D = this.cabD, di = this.cabDi, M = 2047;
          const refl = 0.28 * D[(di - Math.round(0.003 * fs)) & M] + 0.18 * D[(di - Math.round(0.0055 * fs)) & M] + 0.1 * D[(di - Math.round(0.008 * fs)) & M];
          D[di] = c + 0.25 * refl; this.cabDi = (di + 1) & M;
          c += refl;
          eL = (1 - w) * eL + w * c; eR = (1 - w) * eR + w * c * 0.98;
        }
        const st = this.starterLvl > 1e-4 ? this.starter.sample(this.starterLvl * 0.35, this.crank) : 0;
        const ty = (road > 1e-4 || this.skidS > 1e-4) ? this.tyre.sample(road * (1 + 1.2 * w), this.skidS * (1 - 0.35 * w)) : 0;
        const wind = wn * (0.5 + 1.0 * w);
        tL[k] = eL + st * (1 + 0.6 * w) + ty + wind;
        tR[k] = eR + st * (1 + 0.6 * w) + ty * 0.95 + wind * 0.9;
      }
      this.limiter.process(tL, tR, L, R, n);
      this.gr = this.limiter.minG; this.limiter.minG = 1;
    }
  }

  const api = { EngineSynth, EngineVoice, NpRandomState, Starter, TyreNoise, Limiter, bq, lopeOf, DEFAULT_SPEC };

  // ---- AudioWorklet processor (thin wrapper) -------------------------------
  if (typeof registerProcessor === 'function' && typeof AudioWorkletProcessor !== 'undefined') {
    class EngineProcessor extends AudioWorkletProcessor {
      constructor(options) {
        super();
        this.voice = new EngineVoice(sampleRate, (options && options.processorOptions) || {});
        this.port.onmessage = (e) => this.voice.msg(e.data);
        this.n = 0;
        this.port.postMessage({ type: 'hello', fs: sampleRate });
      }
      process(inputs, outputs) {
        const out = outputs[0]; const L = out[0], R = out[1] || out[0];
        this.voice.process(L, R, L.length);
        if ((++this.n & 31) === 0) this.port.postMessage({ type: 'meter', gr: this.voice.gr, blocks: this.n });
        return true;
      }
    }
    registerProcessor('engine-processor', EngineProcessor);
  }
  return api;
})();

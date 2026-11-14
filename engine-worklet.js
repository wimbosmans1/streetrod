(function engineWorkletMain(modules) {
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

  /** v1 engine + the v1.1 in-cabin approximation (built from the synth's stems) */
  class V1Source {
    constructor(spec, fs) {
      this.fs = fs; this.synth = new EngineSynth(spec, fs); this.spec = this.synth.spec;
      const B = 128;
      this.tL = new Float32Array(B); this.tR = new Float32Array(B); this.sE = new Float32Array(B); this.sF = new Float32Array(B);
      // exhaust heard through the body/floor: strong low-pass, cabin boom 40-80 Hz
      this.cabE = [bq('lp', 220, 0.8, fs), bq('lp', 480, 0.7, fs), bq('peak', 58, 2.2, fs, 9), bq('peak', 115, 1.6, fs, 3), bq('hp', 25, 0.7, fs)];
      // intake / engine bay / valvetrain through the firewall: brighter, mid-forward
      this.cabF = [bq('lp', 2600, 0.7, fs), bq('peak', 420, 1.0, fs, 4), bq('peak', 1100, 1.5, fs, 2), bq('hp', 70, 0.7, fs)];
      this.cabD = new Float32Array(2048); this.cabDi = 0;
      this.cabEst = this.cabE.map(() => [0, 0]); this.cabFst = this.cabF.map(() => [0, 0]);
      this.cabGain = 2.2;
    }
    setView() {}
    /** wv[k] = 0 outside .. 1 inside (smoothed by the voice) */
    render(n, rpm, thr, L, R, wv, raw) {
      const tL = this.tL, tR = this.tR, sE = this.sE, sF = this.sF, fs = this.fs;
      this.synth.render(n, rpm, thr, tL, tR, 0, sE, sF);
      for (let k = 0; k < n; k++) {
        let eL = tL[k], eR = tR[k];
        const w = raw ? 0 : wv[k];
        if (w > 1e-3) {
          let ce = sE[k]; for (let j = 0; j < this.cabE.length; j++) ce = bqs(ce, this.cabE[j], this.cabEst[j]);
          let cf = sF[k]; for (let j = 0; j < this.cabF.length; j++) cf = bqs(cf, this.cabF[j], this.cabFst[j]);
          let c = this.cabGain * (0.9 * ce + 0.55 * cf);
          const D = this.cabD, di = this.cabDi, M = 2047;   // short box reflections (~3, 5.5, 8 ms)
          const refl = 0.28 * D[(di - Math.round(0.003 * fs)) & M] + 0.18 * D[(di - Math.round(0.0055 * fs)) & M] + 0.1 * D[(di - Math.round(0.008 * fs)) & M];
          D[di] = c + 0.25 * refl; this.cabDi = (di + 1) & M;
          c += refl;
          eL = (1 - w) * eL + w * c; eR = (1 - w) * eR + w * c * 0.98;
        }
        L[k] = eL; R[k] = eR;
      }
    }
  }

  /** a listener-aware model (v2+): the listener is part of the synth; a view change
   *  clones the synth (same state, same RNG) at the new position and crossfades. */
  class ListenerSource {
    constructor(model, spec, listeners, views, fs, opts) {
      this.model = model; this.fs = fs; this.listeners = listeners; this.views = views || ['rear', 'in_cabin'];
      this.pos = this.views[0];
      this.cur = new model.Synth(spec, listeners[this.pos], fs, Object.assign({}, model.synthOpts || {}, opts)); this.spec = this.cur.spec;
      this.selfGated = !!this.cur.selfGated;      // v3+: the source plays its own start-up / shutdown
      this.g = this.gainOf(this.pos); this.next = null; this.ng = 1; this.xf = 1;
      this.aL = new Float32Array(128); this.aR = new Float32Array(128); this.bL = new Float32Array(128); this.bR = new Float32Array(128);
    }
    ctl(m) { if (this.cur.ctl) this.cur.ctl(m); if (this.next && this.next.ctl) this.next.ctl(m); }
    initRunning(on) { if (this.cur.initRunning) this.cur.initRunning(on); }
    report() { return this.cur.report ? this.cur.report() : null; }
    bay(o) { return this.cur.bay ? this.cur.bay(o) : false; }   // v1.10 engine bay: kernel crank state if the model has one
    gainOf(pos) { const l = this.listeners[pos]; return Math.pow(10, ((l && l.gain_db) || 0) / 20); }
    setView(v) {
      const pos = this.views[v] || this.views[0];
      if (pos === (this.next ? this.npos : this.pos)) return;
      const base = this.next || this.cur;
      if (this.next) { this.cur = this.next; this.pos = this.npos; this.g = this.ng; }
      this.next = base.cloneWithListener(this.listeners[pos]); this.npos = pos; this.ng = this.gainOf(pos); this.xf = 0;
    }
    render(n, rpm, thr, L, R, wv, raw) {
      const gA = raw ? 1 : this.g;
      this.cur.render(n, rpm, thr, this.aL, this.aR, 0);
      if (!this.next) { for (let k = 0; k < n; k++) { L[k] = this.aL[k] * gA; R[k] = this.aR[k] * gA; } return; }
      this.next.render(n, rpm, thr, this.bL, this.bR, 0);
      const step = 1 / (0.35 * this.fs), gB = raw ? 1 : this.ng;
      for (let k = 0; k < n; k++) {
        this.xf = Math.min(1, this.xf + step);
        const b = 0.5 - 0.5 * Math.cos(Math.PI * this.xf), a = 1 - b;    // coherent signals: linear-sum crossfade
        L[k] = a * gA * this.aL[k] + b * gB * this.bL[k]; R[k] = a * gA * this.aR[k] + b * gB * this.bR[k];
      }
      if (this.xf >= 1) { this.cur = this.next; this.pos = this.npos; this.g = this.ng; this.next = null; }
    }
  }

  // v1.10.4: the game's speed wind (band noise 700 Hz-3 kHz, 0.02 x (v/40)^1.5, louder inside) is not engine sound and is off for
  // every car; the code stays (MIXER_WIND = 0.02 brings it back). Tyre / road noise and the starter are unchanged.
  const MIXER_WIND = 0;

  /**
   * EngineVoice: everything the game hears, independent of the audio API, so it
   * runs inside the AudioWorklet or (fallback) a main-thread ScriptProcessor.
   * process(L, R, n) with n <= 128 per call (call repeatedly for bigger buffers).
   * Messages: {type:'spec', model, spec, listeners?, outGain}, {type:'ctl', ...}, {type:'script'}.
   * A spec message (car or sound-model change) crossfades live to the new source.
   */
  class EngineVoice {
    constructor(fs, o) {
      o = o || {};
      this.fs = fs; this.raw = !!o.raw; this.opts = o;
      this.src = null; this.nsrc = null; this.xfade = 1;
      this.target = { rpm: 800, thr: 0 }; this.sm = { rpm: 800, thr: 0 };
      this.smK = 1 - Math.exp(-128 / (0.025 * fs));
      this.engGain = 0; this.engGainTarget = 0; this.gainK = 1 - Math.exp(-1 / (0.03 * fs));
      this.outGain = o.outGain || 1; this.outGainS = this.outGain; this.nOutGain = 1;
      this.starter = new Starter(fs); this.starterLvl = 0; this.starterTarget = 0; this.crank = 200;
      this.tyre = new TyreNoise(fs); this.skid = 0; this.skidS = 0; this.speed = 0; this.speedS = 0;
      this.limiter = new Limiter(fs, -1.0, 3, 180);
      this.script = null; this.scriptPos = 0;
      const B = 128;
      this.tL = new Float32Array(B); this.tR = new Float32Array(B); this.tL2 = new Float32Array(B); this.tR2 = new Float32Array(B); this.wv = new Float32Array(B);
      this.seed = 4242; this.wind = [bq('bp', 700, 0.5, fs), bq('lp', 3000, 0.7, fs)]; this.windSt = [[0, 0], [0, 0]]; this.windLfo = 0;
      this.view = 0; this.listen = 0; this.listenS = 0; this.listenK = 1 - Math.exp(-1 / (0.35 * fs));
      this.blocks = 0; this.gr = 1; this.modelId = '';
      if (o.ctl) this.msg(Object.assign({ type: 'ctl' }, o.ctl));   // (tests) initial controls, before the source exists
      if (o.spec) this.msg({ type: 'spec', model: o.model || 'v1', spec: o.spec, listeners: o.listeners });
      if (o.script) { this.script = o.script; this.engGain = this.engGainTarget = 1; }
    }
    rnd() { this.seed = (Math.imul(this.seed, 1664525) + 1013904223) >>> 0; return this.seed / 4294967296 * 2 - 1; }
    makeSource(m) {
      const model = ENGINES[m.model || 'v1'];
      if (!model) throw new Error('unknown sound model ' + m.model);
      if (model.id === 'v1') return new V1Source(m.spec, this.fs);
      const s = new ListenerSource(model, m.spec, m.listeners || {}, m.views, this.fs, Object.assign({ fir: this.opts.fir || 'fft', ntap: this.opts.ntap }, this.opts.synthOpts || {}));
      s.setView(this.view); if (s.next) { s.cur = s.next; s.pos = s.npos; s.g = s.ng; s.next = null; }
      if (s.selfGated) { s.initRunning(this.engGainTarget > 0); if (this.lastCtl) s.ctl(this.lastCtl); }
      return s;
    }
    msg(m) {
      if (m.type === 'spec') {
        let s;
        try { s = this.makeSource(m); } catch (e) { this.error = String(e && e.message || e); return; }
        s.outGain = m.outGain || 1; s.modelId = m.model || 'v1';
        if (this.src && this.engGain > 0.001) { this.nsrc = s; this.xfade = 0; }
        else { this.src = s; this.nsrc = null; this.outGain = s.outGain; this.modelId = s.modelId; this.sm.rpm = this.target.rpm = s.spec.idle_rpm; s.synth ? (s.synth._rpm = s.spec.idle_rpm) : 0; }
      } else if (m.type === 'ctl') {
        this.lastCtl = Object.assign(this.lastCtl || {}, m);
        if (this.src && this.src.ctl) this.src.ctl(m); if (this.nsrc && this.nsrc.ctl) this.nsrc.ctl(m);
        if (m.rpm !== undefined) this.target.rpm = m.rpm;
        if (m.thr !== undefined) this.target.thr = m.thr;
        if (m.running !== undefined) this.engGainTarget = m.running ? 1 : 0;
        if (m.starter !== undefined) this.starterTarget = m.starter ? 1 : 0;
        if (m.crank !== undefined) this.crank = m.crank;
        if (m.speed !== undefined) this.speed = m.speed;
        if (m.skid !== undefined) this.skid = m.skid;
        if (m.listen !== undefined) {   // view: 0 / 3 rear, 1 / 4 cabin (3–4 are the garage)
          this.view = m.listen | 0; this.listen = (this.view === 1 || this.view === 4) ? 1 : 0;
          if (this.src) this.src.setView(this.listen); if (this.nsrc) this.nsrc.setView(this.listen);
        }
      } else if (m.type === 'script') { this.script = m.script; this.scriptPos = 0; }
      if (m.type === 'ctl' && m.bay !== undefined) this.bayOn = !!m.bay;   // v1.10 engine bay view open: report the crank state
    }
    process(L, R, n) {
      this.blocks++;
      if (!this.src) { for (let k = 0; k < n; k++) { L[k] = 0; R[k] = 0; } return; }
      let rpm, thr;
      if (this.script) {
        const i = Math.min(this.scriptPos, this.script.length - 1);
        rpm = this.script[i][0]; thr = this.script[i][1]; this.scriptPos++;
        if (this.script[i].length > 2 && this.src.ctl) this.src.ctl({ gear: this.script[i][2], tq: this.script[i][3] });   // (tests) gear / torque
      } else {
        this.sm.rpm += (this.target.rpm - this.sm.rpm) * this.smK;
        this.sm.thr += (this.target.thr - this.sm.thr) * this.smK;
        rpm = Math.max(60, this.sm.rpm); thr = Math.min(1, Math.max(0, this.sm.thr));
      }
      const tL = this.tL, tR = this.tR, wv = this.wv;
      for (let k = 0; k < n; k++) { this.listenS += (this.listen - this.listenS) * this.listenK; wv[k] = this.listenS; }
      this.src.render(n, rpm, thr, tL, tR, wv, this.raw);
      // v1.10 engine bay: crank angle of the rpm this source is rendered at (models without their own crank); free when unused
      this.bayRpm = rpm; this.bayTh = ((this.bayTh || 0) + rpm * 6.0 * n / this.fs) % 720.0;
      if (this.raw) { for (let k = 0; k < n; k++) { L[k] = tL[k]; R[k] = tR[k]; } return; }
      // engine on/off gate per source: v1/v2 are gated by the game's running flag (30 ms fade),
      // self-gated sources (v3+) play their own start-up, coast-down and silence
      const ge = this.ge || (this.ge = new Float32Array(128));
      for (let k = 0; k < n; k++) { this.engGain += (this.engGainTarget - this.engGain) * this.gainK; ge[k] = this.engGain; }
      const sgA = !!this.src.selfGated;
      for (let k = 0; k < n; k++) { const g = this.outGain * (sgA ? 1 : ge[k]); tL[k] *= g; tR[k] *= g; }
      if (this.nsrc) {
        this.nsrc.render(n, rpm, thr, this.tL2, this.tR2, wv, false);
        const g2 = this.nsrc.outGain, step = 1 / (0.25 * this.fs), sgB = !!this.nsrc.selfGated;
        for (let k = 0; k < n; k++) {
          this.xfade = Math.min(1, this.xfade + step);
          const a = Math.cos(this.xfade * Math.PI / 2), b = Math.sin(this.xfade * Math.PI / 2), gb = g2 * (sgB ? 1 : ge[k]);
          tL[k] = a * tL[k] + b * gb * this.tL2[k]; tR[k] = a * tR[k] + b * gb * this.tR2[k];
        }
        if (this.xfade >= 1) { this.src = this.nsrc; this.outGain = this.nsrc.outGain; this.modelId = this.nsrc.modelId; this.nsrc = null; }
      }
      const fs = this.fs;
      for (let k = 0; k < n; k++) {
        this.starterLvl += ((this.src.selfGated ? 0 : this.starterTarget) - this.starterLvl) * this.gainK;
        this.speedS += (this.speed - this.speedS) * 0.0005; this.skidS += (this.skid - this.skidS) * 0.002;
        const w = wv[k];
        const eL = tL[k], eR = tR[k];
        const v = this.speedS, vr = v / 40;
        const road = Math.min(0.08, 0.05 * vr * vr);
        let wn = 0;   // wind: broadband, gusty, rising ~ v^3 in power; off since v1.10.4 (MIXER_WIND = 0: not engine sound, every car)
        if (MIXER_WIND > 0 && v > 3) {
          this.windLfo += 2 * Math.PI * 0.3 / fs; if (this.windLfo > 6.2832) this.windLfo -= 6.2832;
          wn = bqs(bqs(this.rnd(), this.wind[0], this.windSt[0]), this.wind[1], this.windSt[1]);
          wn *= MIXER_WIND * Math.pow(vr, 1.5) * (0.85 + 0.15 * Math.sin(this.windLfo));
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

  /** v1.10 engine bay: a thin snapshot of the crank state the audio is rendering (no extra DSP), o = Float32Array(56):
   *  [0] crank angle deg (0..720, 0 = cylinder 1 firing TDC), [1] rpm, [2] throttle, [3] boost (pressure ratio), [4] blower phase (revs),
   *  [5] 1 = full-rate v9 kernel, [6] source kind (9 = v9 kernel, 1 = rpm-integrated), [7] running (engine audible / cranking),
   *  [8..23] per-cylinder combustion energy, [24..39] afterburn energy (v9.9), [40..55] mouth-flame outflow (v9.9) */
  EngineVoice.prototype.bayState = function (o) {
    const s = this.nsrc || this.src;
    let i, own = false;
    for (i = 0; i < 56; i++) o[i] = 0;
    if (s && s.bay) own = s.bay(o);
    const r = s && s.selfGated && s.report ? s.report() : null;
    if (!own) {
      o[0] = this.bayTh || 0; o[2] = this.sm.thr; o[6] = 1;
      o[1] = r ? (r.rpm || 0) : (this.engGainTarget > 0 ? (this.bayRpm || 0) : (this.starterTarget > 0 ? this.crank : 0));
      for (i = 0; i < 16; i++) o[8 + i] = 1;
    }
    o[7] = r ? (r.phase !== 'off' && r.phase !== 'stopped' && r.phase !== 'ign' ? 1 : 0) : ((this.engGainTarget > 0 || this.starterTarget > 0) ? 1 : 0);
    if (!o[7] && !r) o[1] = 0;
    return o;
  };
  /** start-up state of a self-gated source (v3+), for the game's tach / physics hand-over */
  EngineVoice.prototype.report = function () { const s = this.nsrc || this.src; return s && s.selfGated && s.report ? s.report() : null; };
  const api = { EngineSynth, EngineVoice, V1Source, ListenerSource, NpRandomState, Starter, TyreNoise, Limiter, bq, lopeOf, DEFAULT_SPEC };
  // ---- sound-model registry: v1 is built in, later versions are drop-in modules ----
  const ENGINES = { v1: { id: 'v1', Synth: EngineSynth, positions: [] } };
  api.ENGINES = ENGINES;
  (modules || []).forEach((mod) => { const r = mod(api); (Array.isArray(r) ? r : [r]).forEach((m) => { ENGINES[m.id] = m; }); });

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
        if ((++this.n & 31) === 0) { this.port.postMessage({ type: 'meter', gr: this.voice.gr, blocks: this.n, model: this.voice.modelId, err: this.voice.error }); }
        if ((this.n & 7) === 0) { const r = this.voice.report(); if (r || this.lastRep) this.port.postMessage({ type: 'eng', r: r }); this.lastRep = r; }
        if (this.voice.bayOn && (this.n & 7) === 4) { const o = this.voice.bayState(new Float32Array(56)); this.port.postMessage({ type: 'bay', t: currentTime + L.length / sampleRate, s: o }, [o.buffer]); }
        return true;
      }
    }
    registerProcessor('engine-processor', EngineProcessor);
  }
  return api;
})([function engineModuleV2(core) {
  'use strict';
  const bq = core.bq, NpRandomState = core.NpRandomState;
  const NSLOT = 48, DL_LEN = 8192, DL_MASK = DL_LEN - 1, NTAP = 512, CH = 128;
  const RV_COMB = [1116, 1188, 1277, 1356], RV_AP = [556, 441, 579, 464];
  const DL_H0 = 0, DL_P0 = 2, DL_X0 = 4, DL_X1 = 5, DL_C0 = 6, DL_A0 = 10, DL_A2 = 12, NDL = 14;
  const M_EXLL = 0, M_EXLR = 1, M_EXRL = 2, M_EXRR = 3, M_DLL = 4, M_DLR = 5, M_DRL = 6, M_DRR = 7, M_GND = 8, M_GNDD = 9, M_HD = 10, M_IN = 11, M_MK = 12, M_REV = 13, NM = 14;
  const C_AIR = 343.0, C_MUFF = 530.0;
  const DEFAULT_SPEC = {
    name: 'V8', displacement_ci: 289.0, compression: 10.5, idle_rpm: 800.0, redline: 6500.0,
    firing_order: [1, 5, 4, 2, 6, 3, 7, 8], left_bank: [5, 6, 7, 8], cam_duration_050: 220.0, lsa: 114.0,
    lifters: 'solid', carbs: '1x4', carb_cfm: 715.0, air_cleaner: 'open', headers: 'tubular', exhaust: 'sport',
    header_len_m: 0.9, pipe_len_m: 3.0, pipe_len_diff_m: 0.15, pipe_diam_in: 2.25, muffler: '', crossover: 'h',
    hcross: 0.12, richness: 1.0, cyl_imbalance: 0.06, rev_inertia: 1.0, fan_blades: 4, rev_limiter: false, seed: 1,
    k_lope: 1.0, k_drift: 1.0, k_jet: 1.0, k_tl: 1.0, k_intake: 1.0, k_mech: 1.0,
  };
  const DEFAULT_LISTENER = { position: 'rear', distance_m: 1.0, lateral_m: 0.3, ground: 0.7, ambience: 1.0, k_body: 1.0, k_shell: 1.0, k_bay_exh: 1.0 };
  function pick(o, k) { if (!(k in o)) throw new Error('v2: unknown option ' + k); return o[k]; }
  const headerType = (sp) => pick({ tubular: 'tri_y', tri_y: 'tri_y', log: 'log', long_tube: 'long_tube' }, sp.headers);
  const mufflerType = (sp) => sp.muffler ? sp.muffler : pick({ turbo: 'stock_reverse', sport: 'sport_turbo', glasspack: 'glasspack', open: 'none' }, sp.exhaust);
  const openHeaders = (sp) => mufflerType(sp) === 'none' && (sp.pipe_len_m < 0.3 || (!sp.muffler && sp.exhaust === 'open'));
  function lopeIndex(sp) { const l = (sp.cam_duration_050 - 190.0) / 55.0 + (114.0 - sp.lsa) / 16.0; return Math.min(1.6, Math.max(0.0, l)); }

  // ---- radix-2 complex FFT (in place), sign -1 forward, +1 inverse (unscaled) ----
  const fftPlans = {};
  function fftPlan(n) {
    if (fftPlans[n]) return fftPlans[n];
    const lg = Math.round(Math.log2(n)), rev = new Uint32Array(n), cs = new Float64Array(n / 2), sn = new Float64Array(n / 2);
    for (let i = 0; i < n; i++) { let r = 0; for (let b = 0; b < lg; b++) r |= ((i >> b) & 1) << (lg - 1 - b); rev[i] = r; }
    for (let i = 0; i < n / 2; i++) { cs[i] = Math.cos(2 * Math.PI * i / n); sn[i] = Math.sin(2 * Math.PI * i / n); }
    return (fftPlans[n] = { n, rev, cs, sn });
  }
  function fft(re, im, sign) {
    const n = re.length, p = fftPlan(n), rev = p.rev, cs = p.cs, sn = p.sn;
    for (let i = 0; i < n; i++) { const j = rev[i]; if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1, step = n / size;
      for (let s = 0; s < n; s += size) {
        for (let k = 0, t = 0; k < half; k++, t += step) {
          const wr = cs[t], wI = sign < 0 ? -sn[t] : sn[t], a = s + k, b = a + half;
          const xr = re[b] * wr - im[b] * wI, xi = re[b] * wI + im[b] * wr;
          re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi;
        }
      }
    }
  }

  // ---- muffler transmission loss (dB) and minimum-phase FIR design ----
  function tlExpansion(f, m, L) { const k = 2 * Math.PI * f / C_MUFF; const s = Math.sin(k * L); return 10 * Math.log10(1 + 0.25 * (m - 1 / m) * (m - 1 / m) * s * s); }
  function tlHelmholtz(f, f0, depth, q) { q = q || 2.0; const x = (f / f0 - f0 / Math.max(f, 1)) * q; return depth / (1 + x * x); }
  function tlAbsorptive(f, len, packing) { packing = packing === undefined ? 1.0 : packing; return packing * 14.0 * len * (1 - Math.exp(-Math.pow(f / 900.0, 1.2))) / (1 + (f / 9000.0) * (f / 9000.0)); }
  function tlHf(f, dbk, cap) { return Math.min(cap, dbk * Math.max(f - 1000.0, 0) / 1000.0); }
  const MUFFLERS = {
    none: (f) => 0 * f,
    glasspack: (f) => tlAbsorptive(f, 0.56) + 0.5 * tlExpansion(f, 2.0, 0.5),
    cherry_bomb: (f) => tlAbsorptive(f, 0.33) + 0.5 * tlExpansion(f, 2.0, 0.3),
    thrush: (f) => tlAbsorptive(f, 0.45, 1.1) + tlExpansion(f, 2.5, 0.35) * 0.7,
    sport_turbo: (f) => tlExpansion(f, 4.0, 0.32) + tlAbsorptive(f, 0.3) + 2.0 + tlHf(f, 2.0, 8.0),
    stock_reverse: (f) => tlExpansion(f, 6.0, 0.22) + tlExpansion(f, 6.0, 0.37) + tlHelmholtz(f, 95, 8.0) + tlAbsorptive(f, 0.25) + 4.0 + tlHf(f, 5.0, 18.0),
    stock_single: (f) => tlExpansion(f, 5.0, 0.3) + tlExpansion(f, 5.0, 0.18) + 3.0 + tlHf(f, 4.0, 15.0),
  };
  const firCache = {};
  function mufflerFir(kind, fs, ntap, kTl) {
    const key = kind + '|' + fs + '|' + ntap + '|' + kTl;
    if (firCache[key]) return firCache[key];
    const nfft = 8192, nb = nfft / 2 + 1, tlf = pick(MUFFLERS, kind);
    const f = new Float64Array(nb), tl = new Float64Array(nb), lf = new Float64Array(nb);
    const val = 1.0 / (nfft * (1 / fs));
    for (let i = 0; i < nb; i++) { f[i] = i * val; tl[i] = tlf(f[i]) * kTl; lf[i] = Math.log2(Math.max(f[i], 1)); }
    // 1/3-octave smoothing: |log2 f - log2 f_i| <= 1/6 is a contiguous index range (lf is monotone)
    const pre = new Float64Array(nb + 1); for (let i = 0; i < nb; i++) pre[i + 1] = pre[i] + tl[i];
    const sm = new Float64Array(nb); let lo = 0, hi = 0;
    for (let i = 0; i < nb; i++) {
      while (lf[i] - lf[lo] > 1 / 6) lo++;
      if (hi < i) hi = i;
      while (hi + 1 < nb && lf[hi + 1] - lf[i] <= 1 / 6) hi++;
      // f=0 and f<1 share lf=0: Python's mask includes every index with |lf-lf_i|<=1/6
      let l2 = lo; while (l2 > 0 && Math.abs(lf[l2 - 1] - lf[i]) <= 1 / 6) l2--;
      sm[i] = (pre[hi + 1] - pre[l2]) / (hi + 1 - l2);
    }
    const re = new Float64Array(nfft), im = new Float64Array(nfft);
    for (let i = 0; i < nb; i++) re[i] = Math.log(Math.max(Math.pow(10, -sm[i] / 20.0), 1e-6));
    for (let i = 1; i < nfft / 2; i++) re[nfft - i] = re[i];
    fft(re, im, +1); for (let i = 0; i < nfft; i++) { re[i] /= nfft; im[i] = 0; }   // real cepstrum (ifft(...).real)
    for (let i = 1; i < nfft / 2; i++) re[i] *= 2;
    for (let i = nfft / 2 + 1; i < nfft; i++) re[i] = 0;
    fft(re, im, -1);
    for (let i = 0; i < nfft; i++) { const e = Math.exp(re[i]); const a = im[i]; re[i] = e * Math.cos(a); im[i] = e * Math.sin(a); }
    fft(re, im, +1);
    const h = new Float64Array(ntap), M2 = 2 * ntap;
    for (let t = 0; t < ntap; t++) h[t] = re[t] / nfft * (0.5 - 0.5 * Math.cos(2 * Math.PI * (ntap + t) / (M2 - 1)));
    return (firCache[key] = h);
  }

  function chain(specs, fs) { return specs.map((s) => bq(s[0], s[1], s[2], fs, s[3] || 0)); }

  function positionMatrix(sp, lis, fs) {
    const M = new Float64Array(NM), pos = lis.position;
    const shell = pick({ log: 0.12, tri_y: 0.35, long_tube: 0.4 }, headerType(sp));
    let pex, peng, pmas;
    if (pos === 'rear') {
      const d = Math.max(0.3, lis.distance_m), xs = [0.5 - lis.lateral_m, -0.5 - lis.lateral_m], ears = [0.09, -0.09], h = 0.35;
      const gi = [M_EXLL, M_EXLR, M_EXRL, M_EXRR], di = [M_DLL, M_DLR, M_DRL, M_DRR];
      for (let b = 0; b < 2; b++) for (let e = 0; e < 2; e++) {
        const r = Math.sqrt(d * d + Math.pow(xs[b] - ears[e], 2) + Math.pow(0.1, 2));
        M[gi[2 * b + e]] = 1.0 / r; M[di[2 * b + e]] = r / C_AIR * fs;
      }
      const mn = Math.min(M[M_DLL], M[M_DLR], M[M_DRL], M[M_DRR]);
      for (const i of di) M[i] -= mn;
      const r_img = Math.sqrt(d * d + Math.pow(h + 0.4, 2));
      M[M_GND] = -lis.ground * 0.5 * (d / r_img) * 0.0 + lis.ground * (Math.min(d, 3) / r_img) * 0.6;
      M[M_GNDD] = (r_img - d) / C_AIR * fs;
      const g_eng = 0.10 / (1 + 0.3 * d);
      M[M_HD] = 0.25 * shell * g_eng; M[M_IN] = 0.6 * g_eng; M[M_MK] = 0.8 * g_eng;
      M[M_REV] = (0.02 + 0.012 * d) * lis.ambience;
      const air = 16000.0 / (1 + d / 15.0);
      pex = [['lp', air, 0.7]]; peng = [['lp', 1500, 0.7], ['lowshelf', 200, 0.7, 3.0]]; pmas = [['hp', 25, 0.7]];
    } else if (pos === 'engine_bay') {
      M[M_EXLL] = M[M_EXRR] = 0.16; M[M_EXLR] = M[M_EXRL] = 0.12;
      M[M_DLR] = M[M_DRL] = 0.0003 * fs; M[M_GND] = 0.0;
      M[M_HD] = shell * 0.9 * lis.k_shell; M[M_IN] = 1.0; M[M_MK] = 1.0;
      for (const i of [M_EXLL, M_EXRR, M_EXLR, M_EXRL]) M[i] *= lis.k_bay_exh;
      M[M_REV] = 0.03 * lis.ambience;
      pex = [['lp', 1800, 0.7], ['lowshelf', 120, 0.7, -3.0]]; peng = [['hp', 40, 0.7]]; pmas = [['hp', 25, 0.7]];
    } else if (pos === 'in_cabin' || pos === 'in_cabin_down') {
      M[M_EXLL] = M[M_EXRR] = 0.5; M[M_EXLR] = M[M_EXRL] = 0.45;
      M[M_HD] = 0.25 * shell; M[M_IN] = 0.35; M[M_MK] = 0.3; M[M_REV] = 0.0;
      const kb = lis.k_body;
      pex = [['lp', 260 * kb, 0.6], ['lp', 700 * kb, 0.7], ['peak', 48, 2.0, 6.0]];
      peng = [['lp', 450 * kb, 0.7], ['lp', 1200 * kb, 0.7]];
      pmas = [['peak', 95, 2.5, 4.0], ['peak', 140, 3.0, 2.0], ['hp', 22, 0.7]];
      if (pos === 'in_cabin_down') { pex = [['lp', 900, 0.6], ['peak', 48, 2.0, 4.0], ['highshelf', 2000, 0.7, -6]]; peng = [['lp', 900, 0.7]]; }
    } else throw new Error('v2: unknown position ' + pos);
    return { M, pex: chain(pex, fs), peng: chain(peng, fs), pmas: chain(pmas, fs) };
  }
  const zst = (co) => co.map(() => new Float64Array(2));
  function bqr(x, co, st) { const y = co[0] * x + st[0]; st[0] = co[1] * x - co[3] * y + st[1]; st[1] = co[2] * x - co[4] * y; return y; }

  class EngineSynthV2 {
    constructor(specIn, listenerIn, fs, opts) {
      this.spec = Object.assign({}, DEFAULT_SPEC, specIn);
      this.listener = Object.assign({}, DEFAULT_LISTENER, listenerIn || {});
      this.fs = fs || 44100; this.opts = opts || {};
      this.firMode = this.opts.fir || 'fft';      // 'fft' (partitioned) | 'direct'
      this.ntap = this.opts.ntap || NTAP;
      this.listenerAware = true;
      this._build();
      this._rpm = this.spec.idle_rpm; this._thr = 0.0;
    }
    _build() {
      const sp = this.spec, fs = this.fs, lope = lopeIndex(sp), dpc = sp.displacement_ci / 8.0;
      const muf = mufflerType(sp), hdr = headerType(sp);
      const P = {};
      P.fs = fs; P.idle = sp.idle_rpm; P.red = sp.redline;
      P.sig0 = 0.05; P.sigl = (0.06 + 0.24 * lope) * sp.k_lope; P.misf = (0.005 + 0.07 * lope * lope) * sp.k_lope;
      P.huntd = (0.004 + 0.022 * lope) * sp.k_lope; P.hunthz = 0.7 + 0.6 * lope; P.kick = (0.008 + 0.05 * lope) * sp.k_lope;
      P.lopespan = 900.0 + 1300.0 * lope; P.idlemap = 0.30 + 0.10 * lope; P.ovmap = 0.07;
      P.vepk = 2600.0 + 2000.0 * (sp.cam_duration_050 - 190.0) / 60.0; P.vew = 2200.0;
      P.blowdeg = 52.0 + 18.0 * lope; P.wmin = 0.45e-3 * fs; P.wmax = (1.6 + 0.035 * dpc) * 1e-3 * fs;
      P.att = Math.min(0.35, Math.max(0.12, 0.26 - 0.015 * (sp.compression - 8.0)));
      P.amp = (dpc / 36.0) * (0.85 + 0.05 * sp.compression / 10.0);
      const tl_open = pick({ none: 0.42, cherry_bomb: 0.34, glasspack: 0.30, thrush: 0.28, sport_turbo: 0.22, stock_single: 0.12, stock_reverse: 0.10 }, muf);
      P.rasp = tl_open + 0.06 * lope; P.jit = 0.12 + 0.25 * lope;
      P.popr = pick({ none: 0.08, cherry_bomb: 0.07, glasspack: 0.06, thrush: 0.05, sport_turbo: 0.035, stock_single: 0.01, stock_reverse: 0.006 }, muf) * (0.5 + lope);
      P.popa = 1.5;
      P.tick = (sp.lifters === 'solid' ? 1.2 : 0.12) * sp.k_mech;
      const [ncarb, bbl] = pick({ '1x2': [1, 2], '1x4': [1, 4], '2x4': [2, 4], '3x2': [3, 2] }, sp.carbs);
      const clean = pick({ open: 1.0, oilbath: 0.55, none: 1.6 }, sp.air_cleaner);
      P.intg = 0.35 * Math.sqrt(sp.carb_cfm / 600.0) * clean * (1.0 + 0.15 * (ncarb - 1)) * sp.k_intake;
      P.intrev = 0.3 * lope;
      P.secmode = pick({ '1x2': 0, '1x4': 1, '2x4': 1, '3x2': 2 }, sp.carbs);
      P.secboost = pick({ '1x2': 0.0, '1x4': 0.8, '2x4': 1.0, '3x2': 1.4 }, sp.carbs);
      const area = 0.0058 * (sp.carb_cfm / 715.0) * (bbl === 4 ? 0.55 : 1.0);
      const vol = 0.0025 * (sp.displacement_ci / 300.0) / Math.pow(ncarb, 0.3);
      P.hz0 = C_AIR / (2 * Math.PI) * Math.sqrt(area / (vol * 0.15));
      P.hzq = pick({ open: 2.0, oilbath: 1.4, none: 3.0 }, sp.air_cleaner);
      P.mech = (sp.lifters === 'solid' ? 0.0035 : 0.002) * sp.k_mech;
      P.fan = 0.004 * sp.k_mech; P.fanbl = sp.fan_blades;
      const m_coll = pick({ log: 1.6, tri_y: 2.0, long_tube: 3.2 }, hdr);
      P.hr = -(m_coll - 1) / (m_coll + 1) * 1.15;
      P.hdamp = pick({ log: 0.25, tri_y: 0.45, long_tube: 0.5 }, hdr);
      const hl = sp.header_len_m;
      P.hlen = [hl, hl * 1.03]; P.plen = [sp.pipe_len_m, sp.pipe_len_m + sp.pipe_len_diff_m];
      P.openpipe = openHeaders(sp) ? 1.0 : 0.0;
      const a_r = sp.pipe_diam_in * 0.0254 / 2;
      P.pr = -0.85 + 2.0 * a_r; P.pdamp = 0.35;
      P.radc = 1.0 / (1.0 + 2 * Math.PI * 60.0 / fs);
      P.hx = sp.crossover !== 'none' ? sp.hcross : 0.0;
      P.cfm = sp.carb_cfm; P.disp = sp.displacement_ci; P.limit = sp.rev_limiter ? 1.0 : 0.0;
      P.thrk = 1.0 - Math.exp(-1.0 / (0.045 * fs)); P.rich = sp.richness;
      P.jet = pick({ none: 4.0, cherry_bomb: 2.8, glasspack: 2.2, thrush: 2.0, sport_turbo: 1.8, stock_single: 0.7, stock_reverse: 0.6 }, muf) * (1 + 0.3 * lope) * sp.k_jet;
      P.jetk = 1.0 - Math.exp(-1.0 / (0.0006 * fs)); P.jetx = 0.5;
      P.drift = 0.010 * sp.k_drift; P.temp0 = 520.0; P.tempa = 650.0; P.tempk = 1.0 - Math.exp(-1.0 / (1.5 * fs));
      this.P = P;
      const rs = new NpRandomState(sp.seed), fo = sp.firing_order.slice();
      this.ev_bank = new Int32Array(8);
      for (let k = 0; k < 8; k++) this.ev_bank[k] = sp.left_bank.indexOf(fo[k]) >= 0 ? 0 : 1;
      const bias = []; for (let k = 0; k < 8; k++) bias.push(1.0 + sp.cyl_imbalance * rs.randn());
      const m = (((bias[0] + bias[1]) + (bias[2] + bias[3])) + ((bias[4] + bias[5]) + (bias[6] + bias[7]))) / 8;
      this.ev_bias = new Float64Array(bias.map((b) => b / m));
      this.ev_dly = new Float64Array(8);
      if (hdr === 'tri_y' || hdr === 'long_tube') {
        for (let k = 0; k < 8; k++) this.ev_dly[k] = 0.06e-3 * fs * rs.double() * (hdr === 'tri_y' ? 2.0 : 1.0);
      } else {
        const lb = sp.left_bank.slice().sort().join(',');
        for (let k = 0; k < 8; k++) { const c = fo[k]; const pos = lb === '5,6,7,8' ? ((c - 1) % 4) : Math.floor((c - 1) / 2); this.ev_dly[k] = (0.15 + 0.22 * pos) / 520.0 * fs; }
      }
      this.fir = mufflerFir(muf, fs, NTAP, sp.k_tl);
      if (this.ntap < NTAP) this.fir = this.fir.slice(0, this.ntap);
      this._setupFir();
      const JET = {
        none: [['hp', 450, 0.7], ['lp', 7500, 0.7], ['peak', 2500, 1.0, 3.0]],
        cherry_bomb: [['hp', 600, 0.7], ['lp', 5500, 0.7], ['peak', 1700, 1.2, 4.0]],
        glasspack: [['hp', 600, 0.7], ['lp', 5000, 0.7], ['peak', 1500, 1.2, 4.0]],
        thrush: [['hp', 600, 0.7], ['lp', 4800, 0.7], ['peak', 1400, 1.2, 3.5]],
        sport_turbo: [['hp', 700, 0.7], ['lp', 4200, 0.7], ['peak', 1800, 1.2, 3.0]],
        stock_single: [['hp', 500, 0.7], ['lp', 2200, 0.7], ['peak', 1000, 1.0, 0]],
        stock_reverse: [['hp', 500, 0.7], ['lp', 2000, 0.7], ['peak', 900, 1.0, 0]],
      };
      const jet = pick(JET, muf);
      this.jt_co = [chain(jet, fs), chain(jet.map((s) => [s[0], s[1] * 1.05, s[2], s[3]]), fs)];
      this.jt_st = [zst(this.jt_co[0]), zst(this.jt_co[1])];
      this.in_co = chain([['peak', 2 * P.hz0, 2.0, 3.0], ['lp', pick({ open: 3500, oilbath: 1400, none: 6000 }, sp.air_cleaner), 0.7], ['hp', 60, 0.7]], fs);
      this.in_st = zst(this.in_co);
      this.mk_co = chain([['bp', 3300, 9.0], ['bp', 5400, 12.0], ['bp', 1600, 0.8]], fs); this.mk_st = zst(this.mk_co);
      this._setListener(this.listener);
      this.S = { th: 0, thr: 0, lope: 0, hunt: 0, limtog: 0, unb: 0, tex: P.temp0 + P.tempa * P.idlemap * 0.4, drift: 0, ovf: 0, sec: 0,
        nz0: 0, nz1: 0, nz2: 0, hlp0: 0, hlp1: 0, plp0: 0, plp1: 0, rad0: 0, rad1: 0, rady0: 0, rady1: 0, je0: 0, je1: 0, sv1: 0, sv2: 0, fanph: 0 };
      const sx = sp.seed * 7919;
      this.rhi = (0x9E3779B9 ^ Math.floor(sx / 4294967296)) >>> 0;
      this.rlo = (0x7F4A7C15 ^ (sx >>> 0)) >>> 0;
      this.slots = new Float64Array(NSLOT * 6); for (let i = 0; i < NSLOT; i++) this.slots[i * 6] = -1;
      this.dl = new Float64Array(NDL * DL_LEN); this.di = new Int32Array(NDL);
      const sc = fs / 44100.0;
      this.rv_d = RV_COMB.concat(RV_AP).map((d) => Math.trunc(d * sc));
      // pass-1 -> pass-2 buffers
      this.b_yp0 = new Float64Array(CH); this.b_yp1 = new Float64Array(CH); this.b_v0 = new Float64Array(CH); this.b_v1 = new Float64Array(CH);
      this.b_jn0 = new Float64Array(CH); this.b_jn1 = new Float64Array(CH); this.b_hd = new Float64Array(CH); this.b_vi = new Float64Array(CH); this.b_mech = new Float64Array(CH);
      this.rpmA = new Float64Array(CH); this.thrA = new Float64Array(CH);
    }
    _setListener(lis) {
      this.listener = Object.assign({}, DEFAULT_LISTENER, lis);
      const pm = positionMatrix(this.spec, this.listener, this.fs);
      this.M = pm.M; this.pex = pm.pex; this.peng = pm.peng; this.pmas = pm.pmas;
      this.pex_st = [zst(pm.pex), zst(pm.pex)]; this.peng_st = zst(pm.peng); this.pmas_st = [zst(pm.pmas), zst(pm.pmas)];
    }
    _setupFir() {
      const nt = this.fir.length;
      this.hist = new Float64Array(2 * (nt + CH));   // per bank: [nt-1 past samples | chunk]
      if (this.firMode === 'fft') {
        const B = CH, N = 2 * B, np = Math.ceil(nt / B);
        this.np = np; this.Hre = []; this.Him = [];
        for (let p = 0; p < np; p++) {
          const re = new Float64Array(N), im = new Float64Array(N);
          for (let t = 0; t < B && p * B + t < nt; t++) re[t] = this.fir[p * B + t];
          fft(re, im, -1); this.Hre.push(re); this.Him.push(im);
        }
        this.Xre = []; this.Xim = []; for (let p = 0; p < np; p++) { this.Xre.push(new Float64Array(N)); this.Xim.push(new Float64Array(N)); }
        this.xhead = 0; this.prev0 = new Float64Array(B); this.prev1 = new Float64Array(B);
        this.wre = new Float64Array(N); this.wim = new Float64Array(N);
      }
    }
    /** copy of this synth (same state, same RNG stream) heard from another listener position */
    cloneWithListener(lis) {
      const c = Object.create(EngineSynthV2.prototype);
      for (const k of Object.keys(this)) {
        const v = this[k];
        if (ArrayBuffer.isView(v)) c[k] = v.slice();
        else if (Array.isArray(v) && v.length && ArrayBuffer.isView(v[0])) c[k] = v.map((a) => a.slice());
        else if (Array.isArray(v) && v.length && Array.isArray(v[0])) c[k] = v.map((a) => a.map((b) => (ArrayBuffer.isView(b) ? b.slice() : b)));
        else if (v && typeof v === 'object' && !Array.isArray(v) && k === 'S') c[k] = Object.assign({}, v);
        else c[k] = v;
      }
      c._setListener(lis);
      return c;
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
      for (let i = 0; i < NSLOT; i++) { const o = i * 6; if (s[o] < 0) { best = i; break; } if (s[o + 1] > bestpos) { bestpos = s[o + 1]; best = i; } }
      const o = best * 6; s[o] = kind; s[o + 1] = start; s[o + 2] = width; s[o + 3] = att; s[o + 4] = amp; s[o + 5] = nz;
    }
    /** render n frames (rpm/throttle ramp linearly from the previous call's targets, like render()) */
    render(n, rpm, throttle, outL, outR, off) {
      off = off || 0;
      const r0 = this._rpm, t0 = this._thr; this._rpm = rpm; this._thr = throttle;
      for (let s = 0; s < n; s += CH) {
        const m = Math.min(CH, n - s);
        for (let i = 0; i < m; i++) { const tt = (s + i + 1) / n; this.rpmA[i] = r0 + (rpm - r0) * tt; this.thrA[i] = t0 + (throttle - t0) * tt; }
        this._chunk(m, outL, outR, off + s);
      }
    }
    _firBlock(m) {
      const h = this.fir, nt = h.length, H = this.hist, o1 = nt + CH;
      if (this.firMode === 'fft' && m !== CH) this.firMode = 'direct';    // partitioned path needs whole 128 blocks
      if (this.firMode === 'fft') {
        const B = CH, N = 2 * B, np = this.np;
        const xi = (this.xhead + np - 1) % np; this.xhead = xi;            // newest spectrum slot
        const xr = this.Xre[xi], xim = this.Xim[xi], p0 = this.prev0, p1 = this.prev1, y0 = this.b_yp0, y1 = this.b_yp1;
        for (let i = 0; i < B; i++) { xr[i] = p0[i]; xim[i] = p1[i]; xr[B + i] = y0[i]; xim[B + i] = y1[i]; p0[i] = y0[i]; p1[i] = y1[i]; }
        fft(xr, xim, -1);
        const wr = this.wre, wi = this.wim; wr.fill(0); wi.fill(0);
        for (let p = 0; p < np; p++) {
          const X = (xi + p) % np, ar = this.Xre[X], ai = this.Xim[X], hr = this.Hre[p], hi = this.Him[p];
          for (let k = 0; k < N; k++) { wr[k] += ar[k] * hr[k] - ai[k] * hi[k]; wi[k] += ar[k] * hi[k] + ai[k] * hr[k]; }
        }
        fft(wr, wi, +1);
        for (let i = 0; i < B; i++) { this.b_v0[i] = wr[B + i] / N; this.b_v1[i] = wi[B + i] / N; }
        return;
      }
      // direct form, same summation order as Python (newest sample first)
      for (let b = 0; b < 2; b++) {
        const base = b * o1, src = b === 0 ? this.b_yp0 : this.b_yp1, dst = b === 0 ? this.b_v0 : this.b_v1;
        for (let i = 0; i < m; i++) H[base + nt - 1 + i] = src[i];
        for (let i = 0; i < m; i++) {
          let acc = 0.0, j = base + nt - 1 + i;
          for (let t = 0; t < nt; t++) acc += h[t] * H[j - t];
          dst[i] = acc;
        }
        H.copyWithin(base, base + m, base + m + nt - 1);
      }
    }
    _chunk(m, outL, outR, off) {
      const P = this.P, S = this.S, slots = this.slots, dl = this.dl, di = this.di, rv_d = this.rv_d;
      const fs = P.fs, idle = P.idle, red = P.red;
      const hunt_inc = 2.0 * Math.PI * P.hunthz / fs, drift_a = 1.0 / (0.7 * fs);
      const in_co = this.in_co, in_st = this.in_st, mk_co = this.mk_co, mk_st = this.mk_st;
      // ---------------- pass 1 ----------------
      for (let n_ = 0; n_ < m; n_++) {
        const rpm_t = this.rpmA[n_];
        const thr = S.thr + (this.thrA[n_] - S.thr) * P.thrk; S.thr = thr;
        let idle_w = 1.0 - (rpm_t - idle) / P.lopespan;
        if (idle_w < 0.0) idle_w = 0.0; else if (idle_w > 1.0) idle_w = 1.0;
        S.hunt += hunt_inc; if (S.hunt > 2.0 * Math.PI) S.hunt -= 2.0 * Math.PI;
        S.drift += -S.drift * drift_a + P.drift * Math.sqrt(2.0 * drift_a) * this._gauss();
        const hunt = (P.huntd * Math.sin(S.hunt) + S.drift) * idle_w * (1.0 - thr);
        const rpm = rpm_t * (1.0 + S.lope + hunt);
        S.lope *= 0.99985;
        const dth = rpm * 6.0 / fs;
        let ov = 0.0;
        if (thr < 0.06) { ov = (rpm - idle * 1.35) / (idle * 0.9); if (ov < 0.0) ov = 0.0; else if (ov > 1.0) ov = 1.0; ov *= (0.06 - thr) / 0.06; }
        S.ovf = ov;
        let sx;
        if (P.secmode < 0.5) sx = 0.0;
        else if (P.secmode < 1.5) sx = Math.min(1.0, Math.max(0.0, (thr - 0.55) / 0.3)) * Math.min(1.0, Math.max(0.0, (rpm - 2400.0) / 1500.0));
        else sx = Math.min(1.0, Math.max(0.0, (thr - 0.5) / 0.2));
        S.sec += (sx - S.sec) * 0.0004;
        const load_now = P.idlemap + (1.0 - P.idlemap) * thr;
        const ttemp = P.temp0 + P.tempa * load_now * (0.35 + 0.65 * rpm / red) * (1.0 - 0.7 * ov);
        S.tex += (ttemp - S.tex) * P.tempk;
        const th0 = S.th, th1 = th0 + dth;
        const k0 = Math.floor(th0 / 90.0), k1 = Math.floor(th1 / 90.0);
        if (k1 !== k0) {
          const k = ((k1 % 8) + 8) % 8, since = (th1 - k1 * 90.0) / dth, bank = this.ev_bank[k];
          const flow = P.disp * rpm / 3456.0 * 0.9;
          let restr = 1.0;
          if (flow > 1.0) { restr = P.cfm * 1.15 / flow; if (restr > 1.0) restr = 1.0; }
          const mp_open = P.idlemap + (1.0 - P.idlemap) * thr * (0.75 + 0.25 * restr);
          const load = mp_open * (1.0 - ov) + P.ovmap * ov;
          const x = (rpm - P.vepk) / P.vew, ve = 0.75 + 0.25 * Math.exp(-x * x);
          const sig = P.sig0 + P.sigl * idle_w * (1.0 - thr) + 0.5 * ov;
          let a = load * ve * this.ev_bias[k] * (1.0 + sig * this._gauss());
          if (a < 0.02 * load) a = 0.02 * load;
          if (this._rand() < P.misf * idle_w * (1.0 - thr) + 0.08 * ov) { a *= 0.15 + 0.3 * this._rand(); S.unb += 1.0; }
          if (P.limit > 0.5 && rpm_t >= red) { S.limtog = 1.0 - S.limtog; if (S.limtog > 0.5) { a = 0.0; S.unb += 1.0; } }
          S.lope += P.kick * idle_w * (1.0 - thr) * (a / (load * ve * this.ev_bias[k] + 1e-6) - 1.0) * 0.25;
          if (S.lope > 0.08) S.lope = 0.08; else if (S.lope < -0.08) S.lope = -0.08;
          let w = P.blowdeg / dth; if (w > P.wmax) w = P.wmax; if (w < P.wmin) w = P.wmin;
          let att = P.att * (1.3 - 0.85 * load); if (att * w < 5.0) att = 5.0 / w;
          const jit = P.jit * (0.3 + idle_w) * this._gauss();
          const start = since - this.ev_dly[k] - jit * fs * 1e-3;
          const nz = P.rasp * (0.35 + 0.65 * load);
          this._spawn(bank, start, w, att, a * P.amp, nz);
          let wi = 180.0 / dth; if (wi > 0.02 * fs) wi = 0.02 * fs;
          const ia = P.intg * (thr * (0.3 + 0.7 * rpm / red) * (1.0 + P.secboost * S.sec) + P.intrev * idle_w * (1.0 - thr) * (0.6 + 0.8 * this._rand()) + 0.03);
          this._spawn(2, since - 0.25 * wi, wi, 0.35, ia, 1.0);
          const ta = P.tick * (0.4 + 0.6 * rpm / red) * (0.7 + 0.6 * this._rand());
          this._spawn(3, since - 0.3 * 90.0 / dth, 1.0, 0.0, ta, 0.0);
          this._spawn(3, since - 0.7 * 90.0 / dth, 1.0, 0.0, ta * (0.6 + 0.5 * this._rand()), 0.0);
          const pr = P.popr * P.rich * (ov * (0.4 + 0.6 * Math.min(1.0, (rpm - idle) / 2500.0)) + 0.4 * Math.min(S.unb, 3.0) / 3.0 * ov);
          if (this._rand() < pr) {
            const pw = (0.6 + 0.8 * this._rand()) * 1e-3 * fs;
            const pa = P.popa * P.amp * (0.4 + 0.9 * this._rand());
            this._spawn(4.0 + bank, since - (1.0 + 25.0 * this._rand()) * 1e-3 * fs, pw, 0.12, pa, 1.8);
            S.unb = 0.0;
          }
          S.unb *= 0.9;
        }
        S.th = th1 % 720.0;
        // pulses
        let exh0 = 0, exh1 = 0, pop0 = 0, pop1 = 0, intk = 0, tick = 0, je0 = 0, je1 = 0;
        S.nz0 += 0.55 * ((this._rand() * 2.0 - 1.0) - S.nz0);
        S.nz1 += 0.55 * ((this._rand() * 2.0 - 1.0) - S.nz1);
        S.nz2 += 0.35 * ((this._rand() * 2.0 - 1.0) - S.nz2);
        for (let i = 0; i < NSLOT; i++) {
          const o = i * 6, kind = slots[o];
          if (kind < 0) continue;
          const pos = slots[o + 1]; slots[o + 1] = pos + 1.0;
          if (pos < 0.0) continue;
          const ik = Math.trunc(kind);
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
          else if (ik === 2) intk += amp * (0.6 + nzv * S.nz2 * 1.4);
          else if (ik === 4) pop0 += amp * (1.0 + nzv * S.nz0 * 2.0);
          else pop1 += amp * (1.0 + nzv * S.nz1 * 2.0);
        }
        // header waveguides (both banks first: the v2 crossover fix)
        const T = S.tex, c_h = 20.05 * Math.sqrt(T + 120.0), c_p = 20.05 * Math.sqrt(T);
        let yh0 = 0, yh1 = 0;
        for (let b = 0; b < 2; b++) {
          const xin = b === 0 ? exh0 : exh1, line = DL_H0 + b, base = line * DL_LEN, idx = di[line];
          const dh = 2.0 * P.hlen[b] / c_h * fs;
          const dI = Math.trunc(dh), fr = dh - dI, av = dl[base + ((idx - dI) & DL_MASK)], bv = dl[base + ((idx - dI - 1) & DL_MASK)];
          const fbv = av + fr * (bv - av);
          let hs = b === 0 ? S.hlp0 : S.hlp1; hs += P.hdamp * (fbv - hs); if (b === 0) S.hlp0 = hs; else S.hlp1 = hs;
          const yh = xin + P.hr * hs;
          dl[base + idx] = yh; di[line] = (idx + 1) & DL_MASK;
          if (b === 0) yh0 = yh; else yh1 = yh;
        }
        this.b_hd[n_] = yh0 + yh1;
        const hx = P.hx;
        for (let b = 0; b < 2; b++) {
          const yh = b === 0 ? yh0 : yh1, yo = b === 0 ? yh1 : yh0;
          const xp = (1.0 - hx) * yh + hx * yo + (b === 0 ? pop0 : pop1);
          const line = DL_P0 + b, base = line * DL_LEN, idx = di[line];
          let yp;
          if (P.openpipe > 0.5) yp = xp;
          else {
            let dp = 2.0 * P.plen[b] / c_p * fs; if (dp > DL_LEN - 4) dp = DL_LEN - 4;
            const dI = Math.trunc(dp), fr = dp - dI, av = dl[base + ((idx - dI) & DL_MASK)], bv = dl[base + ((idx - dI - 1) & DL_MASK)];
            const fbv = av + fr * (bv - av);
            let ps = b === 0 ? S.plp0 : S.plp1; ps += P.pdamp * (fbv - ps); if (b === 0) S.plp0 = ps; else S.plp1 = ps;
            yp = xp + P.pr * ps;
          }
          dl[base + idx] = yp; di[line] = (idx + 1) & DL_MASK;
          if (b === 0) this.b_yp0[n_] = yp; else this.b_yp1[n_] = yp;
        }
        // jet-noise sources (RNG order: bank 0, bank 1, as in the Python bank loop)
        for (let b = 0; b < 2; b++) {
          const je = b === 0 ? je0 : je1;
          let js = b === 0 ? S.je0 : S.je1; js += P.jetk * (Math.abs(je) - js); if (b === 0) S.je0 = js; else S.je1 = js;
          const ev = js / (P.amp + 1e-9);
          const jn = (this._rand() * 2.0 - 1.0) * ev * Math.pow(ev / 0.35 + 1e-3, P.jetx) * 0.5916 * P.amp * P.jet * (0.6 + 0.6 * rpm / red);
          if (b === 0) this.b_jn0[n_] = jn; else this.b_jn1[n_] = jn;
        }
        // intake: carb Helmholtz (TPT SVF band-pass, time varying)
        let vi = intk;
        const f_h = P.hz0 * (1.0 + 0.35 * S.sec) * Math.sqrt(1.0 + 0.0);
        const g = Math.tan(Math.PI * Math.min(f_h, 0.45 * fs) / fs);
        const kq = 1.0 / (P.hzq * (1.0 - 0.4 * thr));
        const a1 = 1.0 / (1.0 + g * (g + kq));
        const v3 = vi - S.sv2, v1 = a1 * S.sv1 + g * a1 * v3, v2 = S.sv2 + g * v1;
        S.sv1 = 2.0 * v1 - S.sv1; S.sv2 = 2.0 * v2 - S.sv2;
        vi = 0.35 * vi + kq * v1 * 1.6;
        for (let j = 0; j < in_co.length; j++) vi = bqr(vi, in_co[j], in_st[j]);
        this.b_vi[n_] = vi;
        // mechanical: ticks, broadband, fan blade-pass
        const vm = bqr(tick, mk_co[0], mk_st[0]) + 0.6 * bqr(tick, mk_co[1], mk_st[1]);
        const mn = bqr(this._rand() * 2.0 - 1.0, mk_co[2], mk_st[2]) * P.mech * (0.2 + rpm / red);
        S.fanph += 2.0 * Math.PI * P.fanbl * rpm / 60.0 * 1.1 / fs;
        if (S.fanph > 2.0 * Math.PI) S.fanph -= 2.0 * Math.PI;
        const fan = P.fan * Math.pow(rpm / 3000.0, 2) * (Math.sin(S.fanph) * (1.0 + 0.3 * S.nz2) + 0.6 * S.nz2);
        this.b_mech[n_] = vm + mn + fan;
        this.rpmA[n_] = rpm;    // pass 2 needs the modulated rpm (jet already done; kept for reference)
      }
      // ---------------- muffler FIR ----------------
      this._firBlock(m);
      // ---------------- pass 2 ----------------
      const M = this.M, pex = this.pex, peng = this.peng, pmas = this.pmas;
      const jt0 = this.jt_co[0], jt1 = this.jt_co[1], js0 = this.jt_st[0], js1 = this.jt_st[1];
      const r = P.radc;
      const bX0 = DL_X0 * DL_LEN, bX1 = DL_X1 * DL_LEN;
      for (let n_ = 0; n_ < m; n_++) {
        let ys0, ys1;
        {
          const v = this.b_v0[n_]; const yr = r * (S.rady0 + v - S.rad0); S.rad0 = v; S.rady0 = yr;
          let jn = this.b_jn0[n_]; for (let j = 0; j < jt0.length; j++) jn = bqr(jn, jt0[j], js0[j]);
          ys0 = yr + jn;
        }
        {
          const v = this.b_v1[n_]; const yr = r * (S.rady1 + v - S.rad1); S.rad1 = v; S.rady1 = yr;
          let jn = this.b_jn1[n_]; for (let j = 0; j < jt1.length; j++) jn = bqr(jn, jt1[j], js1[j]);
          ys1 = yr + jn;
        }
        let i0 = di[DL_X0]; dl[bX0 + i0] = ys0; i0 = (i0 + 1) & DL_MASK; di[DL_X0] = i0;
        let i1 = di[DL_X1]; dl[bX1 + i1] = ys1; i1 = (i1 + 1) & DL_MASK; di[DL_X1] = i1;
        const rd = (base, idx, d) => { const dI = Math.trunc(d), fr = d - dI, a = dl[base + ((idx - dI) & DL_MASK)], b = dl[base + ((idx - dI - 1) & DL_MASK)]; return a + fr * (b - a); };
        let eL = M[M_EXLL] * rd(bX0, i0, M[M_DLL] + 1.0) + M[M_EXRL] * rd(bX1, i1, M[M_DRL] + 1.0);
        let eR = M[M_EXLR] * rd(bX0, i0, M[M_DLR] + 1.0) + M[M_EXRR] * rd(bX1, i1, M[M_DRR] + 1.0);
        const gsum = 0.5 * (rd(bX0, i0, M[M_DLL] + M[M_GNDD] + 1.0) + rd(bX1, i1, M[M_DRR] + M[M_GNDD] + 1.0));
        eL += M[M_GND] * gsum; eR += M[M_GND] * gsum;
        for (let j = 0; j < pex.length; j++) { eL = bqr(eL, pex[j], this.pex_st[0][j]); eR = bqr(eR, pex[j], this.pex_st[1][j]); }
        let eng = M[M_HD] * this.b_hd[n_] + M[M_IN] * this.b_vi[n_] + M[M_MK] * this.b_mech[n_];
        for (let j = 0; j < peng.length; j++) eng = bqr(eng, peng[j], this.peng_st[j]);
        let L = eL + eng, R = eR + 0.95 * eng;
        for (let j = 0; j < pmas.length; j++) { L = bqr(L, pmas[j], this.pmas_st[0][j]); R = bqr(R, pmas[j], this.pmas_st[1][j]); }
        // ambience
        const mm = 0.5 * (L + R);
        let accr = 0.0;
        for (let c = 0; c < 4; c++) {
          const line = DL_C0 + c, base = line * DL_LEN, idx = di[line];
          const yv = dl[base + ((idx - rv_d[c]) & DL_MASK)];
          dl[base + idx] = mm + 0.62 * yv * 0.85 + 0.0; di[line] = (idx + 1) & DL_MASK;
          accr += yv;
        }
        let wl = accr, wr = accr;
        for (let c = 0; c < 2; c++) {
          let line = DL_A0 + c, base = line * DL_LEN, idx = di[line];
          let bv = dl[base + ((idx - rv_d[4 + c]) & DL_MASK)];
          dl[base + idx] = wl + 0.5 * bv; di[line] = (idx + 1) & DL_MASK; wl = bv - 0.5 * wl;
          line = DL_A2 + c; base = line * DL_LEN; idx = di[line];
          bv = dl[base + ((idx - rv_d[6 + c]) & DL_MASK)];
          dl[base + idx] = wr + 0.5 * bv; di[line] = (idx + 1) & DL_MASK; wr = bv - 0.5 * wr;
        }
        outL[off + n_] = L + M[M_REV] * wl;
        outR[off + n_] = R + M[M_REV] * wr;
      }
    }
  }
  return { id: 'v2', Synth: EngineSynthV2, positions: ['rear', 'engine_bay', 'in_cabin', 'in_cabin_down'], mufflerFir, positionMatrix, fft };
},
function engineModuleV34(core) {
  'use strict';
  const bq = core.bq, NpRandomState = core.NpRandomState;
  const NSLOT = 48, DL_LEN = 8192, DL_MASK = DL_LEN - 1, NTAP = 512, NTAP_IR = 768, CH = 128;
  const RV_COMB = [1116, 1188, 1277, 1356], RV_AP = [556, 441, 579, 464];
  const DL_H0 = 0, DL_P0 = 2, DL_X0 = 4, DL_X1 = 5, DL_C0 = 6, DL_A0 = 10, DL_A2 = 12, NDL = 14;
  const M_EXLL = 0, M_EXLR = 1, M_EXRL = 2, M_EXRR = 3, M_DLL = 4, M_DLR = 5, M_DRL = 6, M_DRR = 7, M_GND = 8, M_GNDD = 9, M_HD = 10, M_IN = 11, M_MK = 12, M_REV = 13, NM = 14;
  const C_AIR = 343.0, C_MUFF = 530.0;
  const DEFAULT_SPEC = {
    name: 'V8', displacement_ci: 289.0, compression: 10.5, idle_rpm: 800.0, redline: 6500.0,
    firing_order: [1, 5, 4, 2, 6, 3, 7, 8], left_bank: [5, 6, 7, 8], cam_duration_050: 220.0, lsa: 114.0,
    lifters: 'solid', carbs: '1x4', carb_cfm: 715.0, air_cleaner: 'open', headers: 'tubular', exhaust: 'sport',
    header_len_m: 0.9, pipe_len_m: 3.0, pipe_len_diff_m: 0.15, pipe_diam_in: 2.25, muffler: '', crossover: 'h',
    hcross: 0.12, richness: 1.0, cyl_imbalance: 0.06, rev_inertia: 1.0, fan_blades: 4, rev_limiter: false, seed: 1,
    k_lope: 1.0, k_drift: 1.0, k_jet: 1.0, k_tl: 1.0, k_intake: 1.0, k_mech: 1.0, afterfire: 1.0, k_ir: 1.0,
  };
  const DEFAULT_LISTENER = { position: 'rear', distance_m: 1.0, lateral_m: 0.3, ground: 0.7, ambience: 1.0, k_body: 1.0, k_shell: 1.0, k_bay_exh: 1.0, darken_db: 0.0, label: '' };
  function pick(o, k) { if (!(k in o)) throw new Error('v34: unknown option ' + k); return o[k]; }
  const headerType = (sp) => pick({ tubular: 'tri_y', tri_y: 'tri_y', log: 'log', long_tube: 'long_tube' }, sp.headers);
  const mufflerType = (sp) => sp.muffler ? sp.muffler : pick({ turbo: 'stock_reverse', sport: 'sport_turbo', glasspack: 'glasspack', open: 'none' }, sp.exhaust);
  const openHeaders = (sp) => mufflerType(sp) === 'none' && (sp.pipe_len_m < 0.3 || (!sp.muffler && sp.exhaust === 'open'));
  function lopeIndex(sp) { const l = (sp.cam_duration_050 - 190.0) / 55.0 + (114.0 - sp.lsa) / 16.0; return Math.min(1.6, Math.max(0.0, l)); }

  // ---- radix-2 complex FFT (in place), sign -1 forward, +1 inverse (unscaled) ----
  const fftPlans = {};
  function fftPlan(n) {
    if (fftPlans[n]) return fftPlans[n];
    const lg = Math.round(Math.log2(n)), rev = new Uint32Array(n), cs = new Float64Array(n / 2), sn = new Float64Array(n / 2);
    for (let i = 0; i < n; i++) { let r = 0; for (let b = 0; b < lg; b++) r |= ((i >> b) & 1) << (lg - 1 - b); rev[i] = r; }
    for (let i = 0; i < n / 2; i++) { cs[i] = Math.cos(2 * Math.PI * i / n); sn[i] = Math.sin(2 * Math.PI * i / n); }
    return (fftPlans[n] = { n, rev, cs, sn });
  }
  function fft(re, im, sign) {
    const n = re.length, p = fftPlan(n), rev = p.rev, cs = p.cs, sn = p.sn;
    for (let i = 0; i < n; i++) { const j = rev[i]; if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1, step = n / size;
      for (let s = 0; s < n; s += size) {
        for (let k = 0, t = 0; k < half; k++, t += step) {
          const wr = cs[t], wI = sign < 0 ? -sn[t] : sn[t], a = s + k, b = a + half;
          const xr = re[b] * wr - im[b] * wI, xi = re[b] * wI + im[b] * wr;
          re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi;
        }
      }
    }
  }

  // ---- muffler transmission loss (dB) and minimum-phase FIR design ----
  function tlExpansion(f, m, L) { const k = 2 * Math.PI * f / C_MUFF; const s = Math.sin(k * L); return 10 * Math.log10(1 + 0.25 * (m - 1 / m) * (m - 1 / m) * s * s); }
  function tlHelmholtz(f, f0, depth, q) { q = q || 2.0; const x = (f / f0 - f0 / Math.max(f, 1)) * q; return depth / (1 + x * x); }
  function tlAbsorptive(f, len, packing) { packing = packing === undefined ? 1.0 : packing; return packing * 14.0 * len * (1 - Math.exp(-Math.pow(f / 900.0, 1.2))) / (1 + (f / 9000.0) * (f / 9000.0)); }
  function tlHf(f, dbk, cap) { return Math.min(cap, dbk * Math.max(f - 1000.0, 0) / 1000.0); }
  const MUFFLERS = {
    none: (f) => 0 * f,
    glasspack: (f) => tlAbsorptive(f, 0.56) + 0.5 * tlExpansion(f, 2.0, 0.5),
    cherry_bomb: (f) => tlAbsorptive(f, 0.33) + 0.5 * tlExpansion(f, 2.0, 0.3),
    thrush: (f) => tlAbsorptive(f, 0.45, 1.1) + tlExpansion(f, 2.5, 0.35) * 0.7,
    sport_turbo: (f) => tlExpansion(f, 4.0, 0.32) + tlAbsorptive(f, 0.3) + 2.0 + tlHf(f, 2.0, 8.0),
    stock_reverse: (f) => tlExpansion(f, 6.0, 0.22) + tlExpansion(f, 6.0, 0.37) + tlHelmholtz(f, 95, 8.0) + tlAbsorptive(f, 0.25) + 4.0 + tlHf(f, 5.0, 18.0),
    stock_single: (f) => tlExpansion(f, 5.0, 0.3) + tlExpansion(f, 5.0, 0.18) + 3.0 + tlHf(f, 4.0, 15.0),
  };
  const firCache = {};
  function mufflerFir(kind, fs, ntap, kTl) {
    const key = kind + '|' + fs + '|' + ntap + '|' + kTl;
    if (firCache[key]) return firCache[key];
    const nfft = 8192, nb = nfft / 2 + 1, tlf = pick(MUFFLERS, kind);
    const f = new Float64Array(nb), tl = new Float64Array(nb), lf = new Float64Array(nb);
    const val = 1.0 / (nfft * (1 / fs));
    for (let i = 0; i < nb; i++) { f[i] = i * val; tl[i] = tlf(f[i]) * kTl; lf[i] = Math.log2(Math.max(f[i], 1)); }
    // 1/3-octave smoothing: |log2 f - log2 f_i| <= 1/6 is a contiguous index range (lf is monotone)
    const pre = new Float64Array(nb + 1); for (let i = 0; i < nb; i++) pre[i + 1] = pre[i] + tl[i];
    const sm = new Float64Array(nb); let lo = 0, hi = 0;
    for (let i = 0; i < nb; i++) {
      while (lf[i] - lf[lo] > 1 / 6) lo++;
      if (hi < i) hi = i;
      while (hi + 1 < nb && lf[hi + 1] - lf[i] <= 1 / 6) hi++;
      // f=0 and f<1 share lf=0: Python's mask includes every index with |lf-lf_i|<=1/6
      let l2 = lo; while (l2 > 0 && Math.abs(lf[l2 - 1] - lf[i]) <= 1 / 6) l2--;
      sm[i] = (pre[hi + 1] - pre[l2]) / (hi + 1 - l2);
    }
    const re = new Float64Array(nfft), im = new Float64Array(nfft);
    for (let i = 0; i < nb; i++) re[i] = Math.log(Math.max(Math.pow(10, -sm[i] / 20.0), 1e-6));
    for (let i = 1; i < nfft / 2; i++) re[nfft - i] = re[i];
    fft(re, im, +1); for (let i = 0; i < nfft; i++) { re[i] /= nfft; im[i] = 0; }   // real cepstrum (ifft(...).real)
    for (let i = 1; i < nfft / 2; i++) re[i] *= 2;
    for (let i = nfft / 2 + 1; i < nfft; i++) re[i] = 0;
    fft(re, im, -1);
    for (let i = 0; i < nfft; i++) { const e = Math.exp(re[i]); const a = im[i]; re[i] = e * Math.cos(a); im[i] = e * Math.sin(a); }
    fft(re, im, +1);
    const h = new Float64Array(ntap), M2 = 2 * ntap;
    for (let t = 0; t < ntap; t++) h[t] = re[t] / nfft * (0.5 - 0.5 * Math.cos(2 * Math.PI * (ntap + t) / (M2 - 1)));
    return (firCache[key] = h);
  }

  function chain(specs, fs) { return specs.map((s) => bq(s[0], s[1], s[2], fs, s[3] || 0)); }

  function positionMatrix(sp, lis, fs) {
    const M = new Float64Array(NM), pos = lis.position;
    const shell = pick({ log: 0.12, tri_y: 0.35, long_tube: 0.4 }, headerType(sp));
    let pex, peng, pmas;
    if (pos === 'rear') {
      const d = Math.max(0.3, lis.distance_m), xs = [0.5 - lis.lateral_m, -0.5 - lis.lateral_m], ears = [0.09, -0.09], h = 0.35;
      const gi = [M_EXLL, M_EXLR, M_EXRL, M_EXRR], di = [M_DLL, M_DLR, M_DRL, M_DRR];
      for (let b = 0; b < 2; b++) for (let e = 0; e < 2; e++) {
        const r = Math.sqrt(d * d + Math.pow(xs[b] - ears[e], 2) + Math.pow(0.1, 2));
        M[gi[2 * b + e]] = 1.0 / r; M[di[2 * b + e]] = r / C_AIR * fs;
      }
      const mn = Math.min(M[M_DLL], M[M_DLR], M[M_DRL], M[M_DRR]);
      for (const i of di) M[i] -= mn;
      const r_img = Math.sqrt(d * d + Math.pow(h + 0.4, 2));
      M[M_GND] = -lis.ground * 0.5 * (d / r_img) * 0.0 + lis.ground * (Math.min(d, 3) / r_img) * 0.6;
      M[M_GNDD] = (r_img - d) / C_AIR * fs;
      const g_eng = 0.10 / (1 + 0.3 * d);
      M[M_HD] = 0.25 * shell * g_eng; M[M_IN] = 0.6 * g_eng; M[M_MK] = 0.8 * g_eng;
      M[M_REV] = (0.02 + 0.012 * d) * lis.ambience;
      const air = 16000.0 / (1 + d / 15.0);
      pex = [['lp', air, 0.7]].concat(lis.darken_db ? [['highshelf', 3000, 0.7, lis.darken_db]] : []); peng = [['lp', 1500, 0.7], ['lowshelf', 200, 0.7, 3.0]]; pmas = [['hp', 25, 0.7]];
    } else if (pos === 'engine_bay') {
      M[M_EXLL] = M[M_EXRR] = 0.16; M[M_EXLR] = M[M_EXRL] = 0.12;
      M[M_DLR] = M[M_DRL] = 0.0003 * fs; M[M_GND] = 0.0;
      M[M_HD] = shell * 0.9 * lis.k_shell; M[M_IN] = 1.0; M[M_MK] = 1.0;
      for (const i of [M_EXLL, M_EXRR, M_EXLR, M_EXRL]) M[i] *= lis.k_bay_exh;
      M[M_REV] = 0.03 * lis.ambience;
      pex = [['lp', 1800, 0.7], ['lowshelf', 120, 0.7, -3.0]]; peng = [['hp', 40, 0.7]]; pmas = [['hp', 25, 0.7]];
    } else if (pos === 'in_cabin' || pos === 'in_cabin_down') {
      M[M_EXLL] = M[M_EXRR] = 0.5; M[M_EXLR] = M[M_EXRL] = 0.45;
      M[M_HD] = 0.25 * shell; M[M_IN] = 0.35; M[M_MK] = 0.3; M[M_REV] = 0.0;
      const kb = lis.k_body;
      pex = [['lp', 260 * kb, 0.6], ['lp', 700 * kb, 0.7], ['peak', 48, 2.0, 6.0]];
      peng = [['lp', 450 * kb, 0.7], ['lp', 1200 * kb, 0.7]];
      pmas = [['peak', 95, 2.5, 4.0], ['peak', 140, 3.0, 2.0], ['hp', 22, 0.7]];
      if (pos === 'in_cabin_down') { pex = [['lp', 900, 0.6], ['peak', 48, 2.0, 4.0], ['highshelf', 2000, 0.7, -6]]; peng = [['lp', 900, 0.7]]; }
    } else throw new Error('v34: unknown position ' + pos);
    return { M, pex: chain(pex, fs), peng: chain(peng, fs), pmas: chain(pmas, fs) };
  }
  const zst = (co) => co.map(() => new Float64Array(2));
  function bqr(x, co, st) { const y = co[0] * x + st[0]; st[0] = co[1] * x - co[3] * y + st[1]; st[1] = co[2] * x - co[4] * y; return y; }

  class EngineSynthV34 {
    constructor(specIn, listenerIn, fs, opts) {
      this.spec = Object.assign({}, DEFAULT_SPEC, specIn);
      this.listener = Object.assign({}, DEFAULT_LISTENER, listenerIn || {});
      this.fs = fs || 44100; this.opts = opts || {}; this.ver = this.opts.ver || 4;
      this.firMode = this.opts.fir || 'fft';      // 'fft' (partitioned) | 'direct'
      this.ntap = this.opts.ntap || 1e9;   // full FIR (512 taps, 768 with the v4 shell IR)
      this.listenerAware = true;
      this._build();
      this._rpm = this.spec.idle_rpm; this._thr = 0.0;
    }
    _build() {
      const sp = this.spec, fs = this.fs, lope = lopeIndex(sp), dpc = sp.displacement_ci / 8.0;
      const muf = mufflerType(sp), hdr = headerType(sp);
      const P = {};
      P.fs = fs; P.idle = sp.idle_rpm; P.red = sp.redline;
      P.sig0 = 0.05; P.sigl = (0.06 + 0.24 * lope) * sp.k_lope; P.misf = (0.005 + 0.07 * lope * lope) * sp.k_lope;
      P.huntd = (0.004 + 0.022 * lope) * sp.k_lope; P.hunthz = 0.7 + 0.6 * lope; P.kick = (0.008 + 0.05 * lope) * sp.k_lope;
      P.lopespan = 900.0 + 1300.0 * lope; P.idlemap = 0.30 + 0.10 * lope; P.ovmap = this.ver >= 4 ? 0.18 : 0.07;
      P.vepk = 2600.0 + 2000.0 * (sp.cam_duration_050 - 190.0) / 60.0; P.vew = 2200.0;
      P.blowdeg = 52.0 + 18.0 * lope; P.wmin = 0.45e-3 * fs; P.wmax = (1.6 + 0.035 * dpc) * 1e-3 * fs;
      P.att = Math.min(0.35, Math.max(0.12, 0.26 - 0.015 * (sp.compression - 8.0)));
      P.amp = (dpc / 36.0) * (0.85 + 0.05 * sp.compression / 10.0);
      const tl_open = pick({ none: 0.42, cherry_bomb: 0.34, glasspack: 0.30, thrush: 0.28, sport_turbo: 0.22, stock_single: 0.12, stock_reverse: 0.10 }, muf);
      P.rasp = tl_open + 0.06 * lope; P.jit = 0.12 + 0.25 * lope;
      P.popr = pick({ none: 0.08, cherry_bomb: 0.07, glasspack: 0.06, thrush: 0.05, sport_turbo: 0.035, stock_single: 0.01, stock_reverse: 0.006 }, muf) * (0.5 + lope);
      P.popa = 1.5;
      const V4 = this.ver >= 4;
      if (V4) {
        P.popr = 0.0; P.popa = 0.0;
        const af = pick(AFTERFIRE, muf);
        P.afon = sp.afterfire; P.afprob = Math.min(0.9, af[0] * (1.0 + 1.0 * Math.max(0.0, lope - 0.3)));
        P.afnmax = af[1] + (lope > 0.7 ? 1.0 : 0.0); P.aflp = 1.0 - Math.exp(-2.0 * Math.PI * af[2] / fs);
        P.afcool = af[3]; P.afthr = 0.35; P.afrpm = Math.max(2.2 * sp.idle_rpm, 1800.0);
        P.afk = 1.0 - Math.exp(-1.0 / (0.30 * fs)); P.loadhk = Math.exp(-1.0 / (2.0 * fs));
        P.loadk = 1.0 - Math.exp(-1.0 / (0.08 * fs)); P.trasp = 0.5; P.tint = 0.6; P.tw = 0.12; P.twoff = 0.35; P.thf = 0.8;
        P.hfc = 1.0 - Math.exp(-2.0 * Math.PI * 1000.0 / fs); P.wander = 0.006 * sp.k_drift; P.onload = 1.0; P.wandk = 1.0 / (3.5 * fs);
      } else { P.afon = 0; P.wander = 0; }
      P.tick = (sp.lifters === 'solid' ? 1.2 : 0.12) * sp.k_mech;
      const [ncarb, bbl] = pick({ '1x2': [1, 2], '1x4': [1, 4], '2x4': [2, 4], '3x2': [3, 2] }, sp.carbs);
      const clean = pick({ open: 1.0, oilbath: 0.55, none: 1.6 }, sp.air_cleaner);
      P.intg = 0.35 * Math.sqrt(sp.carb_cfm / 600.0) * clean * (1.0 + 0.15 * (ncarb - 1)) * sp.k_intake;
      P.intrev = 0.3 * lope;
      P.secmode = pick({ '1x2': 0, '1x4': 1, '2x4': 1, '3x2': 2 }, sp.carbs);
      P.secboost = pick({ '1x2': 0.0, '1x4': 0.8, '2x4': 1.0, '3x2': 1.4 }, sp.carbs);
      const area = 0.0058 * (sp.carb_cfm / 715.0) * (bbl === 4 ? 0.55 : 1.0);
      const vol = 0.0025 * (sp.displacement_ci / 300.0) / Math.pow(ncarb, 0.3);
      P.hz0 = C_AIR / (2 * Math.PI) * Math.sqrt(area / (vol * 0.15));
      P.hzq = pick({ open: 2.0, oilbath: 1.4, none: 3.0 }, sp.air_cleaner);
      P.mech = (sp.lifters === 'solid' ? 0.0035 : 0.002) * sp.k_mech;
      P.fan = 0.004 * sp.k_mech; P.fanbl = sp.fan_blades;
      const m_coll = pick({ log: 1.6, tri_y: 2.0, long_tube: 3.2 }, hdr);
      P.hr = -(m_coll - 1) / (m_coll + 1) * 1.15;
      P.hdamp = pick({ log: 0.25, tri_y: 0.45, long_tube: 0.5 }, hdr);
      const hl = sp.header_len_m;
      P.hlen = [hl, hl * 1.03]; P.plen = [sp.pipe_len_m, sp.pipe_len_m + sp.pipe_len_diff_m];
      P.openpipe = openHeaders(sp) ? 1.0 : 0.0;
      const a_r = sp.pipe_diam_in * 0.0254 / 2;
      P.pr = -0.85 + 2.0 * a_r; P.pdamp = 0.35;
      P.radc = 1.0 / (1.0 + 2 * Math.PI * 60.0 / fs);
      P.hx = sp.crossover !== 'none' ? sp.hcross : 0.0;
      P.cfm = sp.carb_cfm; P.disp = sp.displacement_ci; P.limit = sp.rev_limiter ? 1.0 : 0.0;
      P.thrk = 1.0 - Math.exp(-1.0 / (0.045 * fs)); P.rich = sp.richness;
      P.jet = pick({ none: 4.0, cherry_bomb: 2.8, glasspack: 2.2, thrush: 2.0, sport_turbo: 1.8, stock_single: 0.7, stock_reverse: 0.6 }, muf) * (1 + 0.3 * lope) * sp.k_jet;
      P.jetk = 1.0 - Math.exp(-1.0 / (0.0006 * fs)); P.jetx = 0.5;
      P.drift = 0.010 * sp.k_drift; P.temp0 = 520.0; P.tempa = 650.0; P.tempk = 1.0 - Math.exp(-1.0 / (1.5 * fs));
      P.firep = 1.0; P.pumpa = 0.0; P.crankmod = 0.0;     // v3 start-up controls (defaults = v2 behaviour)
      this.P = P;
      const rs = new NpRandomState(sp.seed), fo = sp.firing_order.slice();
      this.ev_bank = new Int32Array(8);
      for (let k = 0; k < 8; k++) this.ev_bank[k] = sp.left_bank.indexOf(fo[k]) >= 0 ? 0 : 1;
      const bias = []; for (let k = 0; k < 8; k++) bias.push(1.0 + sp.cyl_imbalance * rs.randn());
      const m = (((bias[0] + bias[1]) + (bias[2] + bias[3])) + ((bias[4] + bias[5]) + (bias[6] + bias[7]))) / 8;
      this.ev_bias = new Float64Array(bias.map((b) => b / m));
      this.ev_dly = new Float64Array(8);
      if (hdr === 'tri_y' || hdr === 'long_tube') {
        for (let k = 0; k < 8; k++) this.ev_dly[k] = 0.06e-3 * fs * rs.double() * (hdr === 'tri_y' ? 2.0 : 1.0);
      } else {
        const lb = sp.left_bank.slice().sort().join(',');
        for (let k = 0; k < 8; k++) { const c = fo[k]; const pos = lb === '5,6,7,8' ? ((c - 1) % 4) : Math.floor((c - 1) / 2); this.ev_dly[k] = (0.15 + 0.22 * pos) / 520.0 * fs; }
      }
      this.fir = mufflerFir(muf, fs, NTAP, sp.k_tl);
      if (V4 && sp.k_ir > 0.0) this.fir = mufflerWithIr(this.fir, muf, fs, MUFFLER_WET[muf] * sp.k_ir, NTAP_IR);
      if (V4) {   // afterfire amplitude relative to one combustion pulse AT THE OUTLET (after this muffler)
        const h = this.fir, wp = Math.max(4, Math.trunc(P.wmax)), pulse = new Float64Array(wp);
        for (let i = 0; i < wp; i++) { const s = Math.sin(Math.PI * i / wp); pulse[i] = s * s; }
        let comb = 0;
        for (let n = 0; n < wp + h.length - 1; n++) { let a = 0; const lo = Math.max(0, n - h.length + 1), hi = Math.min(n, wp - 1); for (let i = lo; i <= hi; i++) a += pulse[i] * h[n - i]; comb = Math.max(comb, Math.abs(a)); }
        const np_ = Math.trunc(1e-3 * fs); let st = 0, lpk = 0;
        for (let i = 0; i < np_ + 200; i++) { const x = i < np_ ? Math.pow(Math.sin(Math.PI * i / np_), 2) : 0.0; st += P.aflp * (x - st); lpk = Math.max(lpk, Math.abs(st)); }
        P.afa = AFTERFIRE_AMP * comb / (lpk + 1e-12);
      }
      if (this.ntap < this.fir.length) this.fir = this.fir.slice(0, this.ntap);
      this._setupFir();
      const JET = {
        none: [['hp', 450, 0.7], ['lp', 7500, 0.7], ['peak', 2500, 1.0, 3.0]],
        cherry_bomb: [['hp', 600, 0.7], ['lp', 5500, 0.7], ['peak', 1700, 1.2, 4.0]],
        glasspack: [['hp', 600, 0.7], ['lp', 5000, 0.7], ['peak', 1500, 1.2, 4.0]],
        thrush: [['hp', 600, 0.7], ['lp', 4800, 0.7], ['peak', 1400, 1.2, 3.5]],
        sport_turbo: [['hp', 700, 0.7], ['lp', 4200, 0.7], ['peak', 1800, 1.2, 3.0]],
        stock_single: [['hp', 500, 0.7], ['lp', 2200, 0.7], ['peak', 1000, 1.0, 0]],
        stock_reverse: [['hp', 500, 0.7], ['lp', 2000, 0.7], ['peak', 900, 1.0, 0]],
      };
      const jet = pick(JET, muf);
      this.jt_co = [chain(jet, fs), chain(jet.map((s) => [s[0], s[1] * 1.05, s[2], s[3]]), fs)];
      this.jt_st = [zst(this.jt_co[0]), zst(this.jt_co[1])];
      this.in_co = chain([['peak', 2 * P.hz0, 2.0, 3.0], ['lp', pick({ open: 3500, oilbath: 1400, none: 6000 }, sp.air_cleaner), 0.7], ['hp', 60, 0.7]], fs);
      this.in_st = zst(this.in_co);
      this.mk_co = chain([['bp', 3300, 9.0], ['bp', 5400, 12.0], ['bp', 1600, 0.8]], fs); this.mk_st = zst(this.mk_co);
      this._setListener(this.listener);
      this.S = { th: 0, thr: 0, lope: 0, hunt: 0, limtog: 0, unb: 0, tex: P.temp0 + P.tempa * P.idlemap * 0.4, drift: 0, ovf: 0, sec: 0,
        nz0: 0, nz1: 0, nz2: 0, hlp0: 0, hlp1: 0, plp0: 0, plp1: 0, rad0: 0, rad1: 0, rady0: 0, rady1: 0, je0: 0, je1: 0, sv1: 0, sv2: 0, fanph: 0, nfire: 0, nevt: 0,
        tslow: 0, loadh: 0, afcd: 0, afburst: 0, afnext: 0, aflp0: 0, aflp1: 0, afcnt: 0, lu: 0, hflp0: 0, hflp1: 0, wander: 0, aflifts: 0 };
      const sx = sp.seed * 7919;
      this.rhi = (0x9E3779B9 ^ Math.floor(sx / 4294967296)) >>> 0;
      this.rlo = (0x7F4A7C15 ^ (sx >>> 0)) >>> 0;
      this.slots = new Float64Array(NSLOT * 6); for (let i = 0; i < NSLOT; i++) this.slots[i * 6] = -1;
      this.dl = new Float64Array(NDL * DL_LEN); this.di = new Int32Array(NDL);
      const sc = fs / 44100.0;
      this.rv_d = RV_COMB.concat(RV_AP).map((d) => Math.trunc(d * sc));
      // pass-1 -> pass-2 buffers
      this.b_yp0 = new Float64Array(CH); this.b_yp1 = new Float64Array(CH); this.b_v0 = new Float64Array(CH); this.b_v1 = new Float64Array(CH);
      this.b_jn0 = new Float64Array(CH); this.b_jn1 = new Float64Array(CH); this.b_hd = new Float64Array(CH); this.b_vi = new Float64Array(CH); this.b_mech = new Float64Array(CH);
      this.rpmA = new Float64Array(CH); this.thrA = new Float64Array(CH);
      this.b_lu = new Float64Array(CH); this.b_pop0 = new Float64Array(CH); this.b_pop1 = new Float64Array(CH); this.aux = new Float64Array(256);
    }
    _setListener(lis) {
      this.listener = Object.assign({}, DEFAULT_LISTENER, lis);
      const pm = positionMatrix(this.spec, this.listener, this.fs);
      this.M = pm.M; this.pex = pm.pex; this.peng = pm.peng; this.pmas = pm.pmas;
      this.pex_st = [zst(pm.pex), zst(pm.pex)]; this.peng_st = zst(pm.peng); this.pmas_st = [zst(pm.pmas), zst(pm.pmas)];
    }
    _setupFir() {
      const nt = this.fir.length;
      this.hist = new Float64Array(2 * (nt + CH));   // per bank: [nt-1 past samples | chunk]
      if (this.firMode === 'fft') {
        const B = CH, N = 2 * B, np = Math.ceil(nt / B);
        this.np = np; this.Hre = []; this.Him = [];
        for (let p = 0; p < np; p++) {
          const re = new Float64Array(N), im = new Float64Array(N);
          for (let t = 0; t < B && p * B + t < nt; t++) re[t] = this.fir[p * B + t];
          fft(re, im, -1); this.Hre.push(re); this.Him.push(im);
        }
        this.Xre = []; this.Xim = []; for (let p = 0; p < np; p++) { this.Xre.push(new Float64Array(N)); this.Xim.push(new Float64Array(N)); }
        this.xhead = 0; this.prev0 = new Float64Array(B); this.prev1 = new Float64Array(B);
        this.wre = new Float64Array(N); this.wim = new Float64Array(N);
      }
    }
    /** copy of this synth (same state, same RNG stream) heard from another listener position */
    cloneWithListener(lis) {
      const c = Object.create(EngineSynthV34.prototype);
      for (const k of Object.keys(this)) {
        const v = this[k];
        if (ArrayBuffer.isView(v)) c[k] = v.slice();
        else if (Array.isArray(v) && v.length && ArrayBuffer.isView(v[0])) c[k] = v.map((a) => a.slice());
        else if (Array.isArray(v) && v.length && Array.isArray(v[0])) c[k] = v.map((a) => a.map((b) => (ArrayBuffer.isView(b) ? b.slice() : b)));
        else if (v && typeof v === 'object' && !Array.isArray(v) && k === 'S') c[k] = Object.assign({}, v);
        else c[k] = v;
      }
      c._setListener(lis);
      return c;
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
      for (let i = 0; i < NSLOT; i++) { const o = i * 6; if (s[o] < 0) { best = i; break; } if (s[o + 1] > bestpos) { bestpos = s[o + 1]; best = i; } }
      const o = best * 6; s[o] = kind; s[o + 1] = start; s[o + 2] = width; s[o + 3] = att; s[o + 4] = amp; s[o + 5] = nz;
    }
    /** render n frames (rpm/throttle ramp linearly from the previous call's targets, like render()) */
    render(n, rpm, throttle, outL, outR, off) {
      off = off || 0;
      if (this.aux.length < n) this.aux = new Float64Array(n);
      const r0 = this._rpm, t0 = this._thr; this._rpm = rpm; this._thr = throttle;
      for (let s = 0; s < n; s += CH) {
        const m = Math.min(CH, n - s);
        for (let i = 0; i < m; i++) { const tt = (s + i + 1) / n; this.rpmA[i] = r0 + (rpm - r0) * tt; this.thrA[i] = t0 + (throttle - t0) * tt; }
        this._chunk(m, outL, outR, off + s);
        for (let i = 0; i < m; i++) this.aux[s + i] = this.rpmA[i];
      }
    }
    _firBlock(m) {
      const h = this.fir, nt = h.length, H = this.hist, o1 = nt + CH;
      if (this.firMode === 'fft' && m !== CH) this.firMode = 'direct';    // partitioned path needs whole 128 blocks
      if (this.firMode === 'fft') {
        const B = CH, N = 2 * B, np = this.np;
        const xi = (this.xhead + np - 1) % np; this.xhead = xi;            // newest spectrum slot
        const xr = this.Xre[xi], xim = this.Xim[xi], p0 = this.prev0, p1 = this.prev1, y0 = this.b_yp0, y1 = this.b_yp1;
        for (let i = 0; i < B; i++) { xr[i] = p0[i]; xim[i] = p1[i]; xr[B + i] = y0[i]; xim[B + i] = y1[i]; p0[i] = y0[i]; p1[i] = y1[i]; }
        fft(xr, xim, -1);
        const wr = this.wre, wi = this.wim; wr.fill(0); wi.fill(0);
        for (let p = 0; p < np; p++) {
          const X = (xi + p) % np, ar = this.Xre[X], ai = this.Xim[X], hr = this.Hre[p], hi = this.Him[p];
          for (let k = 0; k < N; k++) { wr[k] += ar[k] * hr[k] - ai[k] * hi[k]; wi[k] += ar[k] * hi[k] + ai[k] * hr[k]; }
        }
        fft(wr, wi, +1);
        for (let i = 0; i < B; i++) { this.b_v0[i] = wr[B + i] / N; this.b_v1[i] = wi[B + i] / N; }
        return;
      }
      // direct form, same summation order as Python (newest sample first)
      for (let b = 0; b < 2; b++) {
        const base = b * o1, src = b === 0 ? this.b_yp0 : this.b_yp1, dst = b === 0 ? this.b_v0 : this.b_v1;
        for (let i = 0; i < m; i++) H[base + nt - 1 + i] = src[i];
        for (let i = 0; i < m; i++) {
          let acc = 0.0, j = base + nt - 1 + i;
          for (let t = 0; t < nt; t++) acc += h[t] * H[j - t];
          dst[i] = acc;
        }
        H.copyWithin(base, base + m, base + m + nt - 1);
      }
    }
    _chunk(m, outL, outR, off) {
      const P = this.P, S = this.S, slots = this.slots, dl = this.dl, di = this.di, rv_d = this.rv_d;
      const fs = P.fs, idle = P.idle, red = P.red;
      const hunt_inc = 2.0 * Math.PI * P.hunthz / fs, drift_a = 1.0 / (0.7 * fs);
      const in_co = this.in_co, in_st = this.in_st, mk_co = this.mk_co, mk_st = this.mk_st, V4 = this.ver >= 4;
      // ---------------- pass 1 ----------------
      for (let n_ = 0; n_ < m; n_++) {
        const rpm_t = this.rpmA[n_];
        const thr = S.thr + (this.thrA[n_] - S.thr) * P.thrk; S.thr = thr;
        let idle_w = 1.0 - (rpm_t - idle) / P.lopespan;
        if (idle_w < 0.0) idle_w = 0.0; else if (idle_w > 1.0) idle_w = 1.0;
        S.hunt += hunt_inc; if (S.hunt > 2.0 * Math.PI) S.hunt -= 2.0 * Math.PI;
        S.drift += -S.drift * drift_a + P.drift * Math.sqrt(2.0 * drift_a) * this._gauss();
        if (P.wander > 0.0) S.wander += -S.wander * P.wandk + P.wander * Math.sqrt(2.0 * P.wandk) * this._gauss();
        const hunt = (P.huntd * Math.sin(S.hunt) + S.drift + S.wander) * idle_w * (1.0 - thr);
        let rpm = rpm_t * (1.0 + S.lope + hunt);
        if (P.crankmod > 0.0) {   // v3 cranking / run-down: crank slows into each compression stroke
          const ph = (S.th % 90.0) / 90.0, kc = (Math.floor(S.th / 90.0) + 1) % 8;
          rpm *= 1.0 + P.crankmod * this.ev_bias[kc] * Math.cos(2.0 * Math.PI * (ph - 0.4));
          if (rpm < 0.0) rpm = 0.0;
        }
        S.lope *= 0.99985;
        const dth = rpm * 6.0 / fs;
        let ov = 0.0;
        if (thr < 0.06) { ov = (rpm - idle * 1.35) / (idle * 0.9); if (ov < 0.0) ov = 0.0; else if (ov > 1.0) ov = 1.0; ov *= (0.06 - thr) / 0.06; }
        S.ovf = ov;
        let lu = 0.0;
        if (V4) {
          // v4 on/off-load timbre state: +1 full load, 0 idle / steady light, -1 hard overrun
          S.lu += ((thr - ov) - S.lu) * P.loadk;
          lu = S.lu < 0.0 ? S.lu : S.lu * P.onload;
          // v4 gated afterfire: one roll per fast lift from recent high load, then a cooldown
          S.tslow += (thr - S.tslow) * P.afk;
          const lh = thr * Math.min(1.0, rpm / (0.5 * red));
          if (lh > S.loadh) S.loadh = lh; else S.loadh *= P.loadhk;
          if (S.afcd > 0.0) S.afcd -= 1.0;
          if (P.afon > 0.0 && S.afcd <= 0.0 && (S.tslow - thr) > P.afthr && S.loadh > 0.55 && rpm > P.afrpm) {
            S.afcd = P.afcool * fs * (0.8 + 0.4 * this._rand());
            S.aflifts += 1.0;
            let egt = (S.tex - 600.0) / 300.0; if (egt < 0.0) egt = 0.0; else if (egt > 1.0) egt = 1.0;
            let pf = P.afprob * P.afon * egt * P.rich; if (pf > 0.9) pf = 0.9;
            if (this._rand() < pf) { S.afburst = 1.0 + Math.trunc(this._rand() * P.afnmax); S.afnext = (0.015 + 0.06 * this._rand()) * fs; }
          }
          if (S.afburst > 0.0) {
            if (thr > S.tslow + 0.05 || rpm < idle * 1.4) S.afburst = 0.0;
            else {
              S.afnext -= 1.0;
              if (S.afnext <= 0.0) {
                const pw = (0.6 + 0.8 * this._rand()) * 1e-3 * fs;
                const pa = P.afa * P.amp * (0.5 + 0.5 * this._rand());
                this._spawn(4.0 + (this._rand() < 0.5 ? 1.0 : 0.0), 0.0, pw, 0.12, pa, 1.8);
                S.afburst -= 1.0; S.afcnt += 1.0; S.afnext = (0.04 + 0.14 * this._rand()) * fs;
              }
            }
          }
        }
        this.b_lu[n_] = lu;
        let sx;
        if (P.secmode < 0.5) sx = 0.0;
        else if (P.secmode < 1.5) sx = Math.min(1.0, Math.max(0.0, (thr - 0.55) / 0.3)) * Math.min(1.0, Math.max(0.0, (rpm - 2400.0) / 1500.0));
        else sx = Math.min(1.0, Math.max(0.0, (thr - 0.5) / 0.2));
        S.sec += (sx - S.sec) * 0.0004;
        const load_now = P.idlemap + (1.0 - P.idlemap) * thr;
        const ttemp = P.temp0 + P.tempa * load_now * (0.35 + 0.65 * rpm / red) * (1.0 - 0.7 * ov);
        S.tex += (ttemp - S.tex) * P.tempk;
        const th0 = S.th, th1 = th0 + dth;
        const k0 = Math.floor(th0 / 90.0), k1 = Math.floor(th1 / 90.0);
        if (k1 !== k0) {
          const k = ((k1 % 8) + 8) % 8, since = (th1 - k1 * 90.0) / dth, bank = this.ev_bank[k];
          const flow = P.disp * rpm / 3456.0 * 0.9;
          let restr = 1.0;
          if (flow > 1.0) { restr = P.cfm * 1.15 / flow; if (restr > 1.0) restr = 1.0; }
          const mp_open = P.idlemap + (1.0 - P.idlemap) * thr * (0.75 + 0.25 * restr);
          const load = mp_open * (1.0 - ov) + P.ovmap * ov;
          const x = (rpm - P.vepk) / P.vew, ve = 0.75 + 0.25 * Math.exp(-x * x);
          const sig = P.sig0 + P.sigl * idle_w * (1.0 - thr) + 0.5 * ov;
          let a = load * ve * this.ev_bias[k] * (1.0 + sig * this._gauss());
          if (a < 0.02 * load) a = 0.02 * load;
          // v3: ignition / fuel availability (start-up, shutdown). firep == 1 -> v2 path, same RNG sequence
          let unfired = false;
          if (P.firep < 1.0) { if (this._rand() >= P.firep) unfired = true; }
          S.nevt += 1.0;
          if (unfired) a = P.pumpa * this.ev_bias[k] * (0.8 + 0.4 * this._rand());   // pumping puff only
          else {
            S.nfire += 1.0;
            if (this._rand() < P.misf * idle_w * (1.0 - thr) + 0.08 * ov) { a *= 0.15 + 0.3 * this._rand(); S.unb += 1.0; }
            if (P.limit > 0.5 && rpm_t >= red) { S.limtog = 1.0 - S.limtog; if (S.limtog > 0.5) { a = 0.0; S.unb += 1.0; } }
            S.lope += P.kick * idle_w * (1.0 - thr) * (a / (load * ve * this.ev_bias[k] + 1e-6) - 1.0) * 0.25;
            if (S.lope > 0.08) S.lope = 0.08; else if (S.lope < -0.08) S.lope = -0.08;
          }
          let w = P.blowdeg / dth; if (w > P.wmax) w = P.wmax; if (w < P.wmin) w = P.wmin;
          let att;
          if (V4) { w *= 1.0 - (lu > 0.0 ? P.tw * lu : P.twoff * lu); att = P.att * (1.3 - 0.85 * load) * (1.0 - 0.5 * P.tw * lu); }
          else att = P.att * (1.3 - 0.85 * load);
          if (att * w < 5.0) att = 5.0 / w;
          const jit = P.jit * (0.3 + idle_w) * this._gauss();
          const start = since - this.ev_dly[k] - jit * fs * 1e-3;
          const nz = V4 ? P.rasp * (0.35 + 0.65 * load) * Math.max(0.0, 1.0 + P.trasp * lu) : P.rasp * (0.35 + 0.65 * load);
          this._spawn(bank, start, w, att, a * P.amp, nz);
          let wi = 180.0 / dth; if (wi > 0.02 * fs) wi = 0.02 * fs;
          let ia = P.intg * (thr * (0.3 + 0.7 * rpm / red) * (1.0 + P.secboost * S.sec) + P.intrev * idle_w * (1.0 - thr) * (0.6 + 0.8 * this._rand()) + 0.03);
          if (V4) ia *= Math.max(0.1, 1.0 + P.tint * lu);
          this._spawn(2, since - 0.25 * wi, wi, 0.35, ia, 1.0);
          const ta = P.tick * (0.4 + 0.6 * rpm / red) * (0.7 + 0.6 * this._rand());
          this._spawn(3, since - 0.3 * 90.0 / dth, 1.0, 0.0, ta, 0.0);
          this._spawn(3, since - 0.7 * 90.0 / dth, 1.0, 0.0, ta * (0.6 + 0.5 * this._rand()), 0.0);
          const pr = P.popr * P.rich * (ov * (0.4 + 0.6 * Math.min(1.0, (rpm - idle) / 2500.0)) + 0.4 * Math.min(S.unb, 3.0) / 3.0 * ov);
          if (V4) { if (!unfired) this._rand(); }      // v4: v2 pop roll removed, one dummy draw keeps the sequence
          else if (!unfired && this._rand() < pr) {
            const pw = (0.6 + 0.8 * this._rand()) * 1e-3 * fs;
            const pa = P.popa * P.amp * (0.4 + 0.9 * this._rand());
            this._spawn(4.0 + bank, since - (1.0 + 25.0 * this._rand()) * 1e-3 * fs, pw, 0.12, pa, 1.8);
            S.unb = 0.0;
          }
          S.unb *= 0.9;
        }
        S.th = th1 % 720.0;
        // pulses
        let exh0 = 0, exh1 = 0, pop0 = 0, pop1 = 0, intk = 0, tick = 0, je0 = 0, je1 = 0;
        S.nz0 += 0.55 * ((this._rand() * 2.0 - 1.0) - S.nz0);
        S.nz1 += 0.55 * ((this._rand() * 2.0 - 1.0) - S.nz1);
        S.nz2 += 0.35 * ((this._rand() * 2.0 - 1.0) - S.nz2);
        for (let i = 0; i < NSLOT; i++) {
          const o = i * 6, kind = slots[o];
          if (kind < 0) continue;
          const pos = slots[o + 1]; slots[o + 1] = pos + 1.0;
          if (pos < 0.0) continue;
          const ik = Math.trunc(kind);
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
          else if (ik === 2) intk += amp * (0.6 + nzv * S.nz2 * 1.4);
          else if (ik === 4) pop0 += amp * (1.0 + nzv * S.nz0 * 2.0);
          else pop1 += amp * (1.0 + nzv * S.nz1 * 2.0);
        }
        // header waveguides (both banks first: the v2 crossover fix)
        const T = S.tex, c_h = 20.05 * Math.sqrt(T + 120.0), c_p = 20.05 * Math.sqrt(T);
        let yh0 = 0, yh1 = 0;
        for (let b = 0; b < 2; b++) {
          const xin = b === 0 ? exh0 : exh1, line = DL_H0 + b, base = line * DL_LEN, idx = di[line];
          const dh = 2.0 * P.hlen[b] / c_h * fs;
          const dI = Math.trunc(dh), fr = dh - dI, av = dl[base + ((idx - dI) & DL_MASK)], bv = dl[base + ((idx - dI - 1) & DL_MASK)];
          const fbv = av + fr * (bv - av);
          let hs = b === 0 ? S.hlp0 : S.hlp1; hs += P.hdamp * (fbv - hs); if (b === 0) S.hlp0 = hs; else S.hlp1 = hs;
          const yh = xin + P.hr * hs;
          dl[base + idx] = yh; di[line] = (idx + 1) & DL_MASK;
          if (b === 0) yh0 = yh; else yh1 = yh;
        }
        this.b_hd[n_] = yh0 + yh1;
        const hx = P.hx;
        for (let b = 0; b < 2; b++) {
          const yh = b === 0 ? yh0 : yh1, yo = b === 0 ? yh1 : yh0;
          const xp = (1.0 - hx) * yh + hx * yo + (V4 ? 0.0 : (b === 0 ? pop0 : pop1));
          const line = DL_P0 + b, base = line * DL_LEN, idx = di[line];
          let yp;
          if (P.openpipe > 0.5) yp = xp;
          else {
            let dp = 2.0 * P.plen[b] / c_p * fs; if (dp > DL_LEN - 4) dp = DL_LEN - 4;
            const dI = Math.trunc(dp), fr = dp - dI, av = dl[base + ((idx - dI) & DL_MASK)], bv = dl[base + ((idx - dI - 1) & DL_MASK)];
            const fbv = av + fr * (bv - av);
            let ps = b === 0 ? S.plp0 : S.plp1; ps += P.pdamp * (fbv - ps); if (b === 0) S.plp0 = ps; else S.plp1 = ps;
            yp = xp + P.pr * ps;
          }
          dl[base + idx] = yp; di[line] = (idx + 1) & DL_MASK;
          if (b === 0) this.b_yp0[n_] = yp; else this.b_yp1[n_] = yp;
        }
        // jet-noise sources (RNG order: bank 0, bank 1, as in the Python bank loop)
        for (let b = 0; b < 2; b++) {
          const je = b === 0 ? je0 : je1;
          let js = b === 0 ? S.je0 : S.je1; js += P.jetk * (Math.abs(je) - js); if (b === 0) S.je0 = js; else S.je1 = js;
          const ev = js / (P.amp + 1e-9);
          const jn = (this._rand() * 2.0 - 1.0) * ev * Math.pow(ev / 0.35 + 1e-3, P.jetx) * 0.5916 * P.amp * P.jet * (0.6 + 0.6 * rpm / red);
          if (b === 0) this.b_jn0[n_] = jn; else this.b_jn1[n_] = jn;
        }
        // intake: carb Helmholtz (TPT SVF band-pass, time varying)
        let vi = intk;
        const f_h = P.hz0 * (1.0 + 0.35 * S.sec) * Math.sqrt(1.0 + 0.0);
        const g = Math.tan(Math.PI * Math.min(f_h, 0.45 * fs) / fs);
        const kq = 1.0 / (P.hzq * (1.0 - 0.4 * thr));
        const a1 = 1.0 / (1.0 + g * (g + kq));
        const v3 = vi - S.sv2, v1 = a1 * S.sv1 + g * a1 * v3, v2 = S.sv2 + g * v1;
        S.sv1 = 2.0 * v1 - S.sv1; S.sv2 = 2.0 * v2 - S.sv2;
        vi = 0.35 * vi + kq * v1 * 1.6;
        for (let j = 0; j < in_co.length; j++) vi = bqr(vi, in_co[j], in_st[j]);
        this.b_vi[n_] = vi;
        // mechanical: ticks, broadband, fan blade-pass
        const vm = bqr(tick, mk_co[0], mk_st[0]) + 0.6 * bqr(tick, mk_co[1], mk_st[1]);
        const mn = bqr(this._rand() * 2.0 - 1.0, mk_co[2], mk_st[2]) * P.mech * (0.2 + rpm / red);
        S.fanph += 2.0 * Math.PI * P.fanbl * rpm / 60.0 * 1.1 / fs;
        if (S.fanph > 2.0 * Math.PI) S.fanph -= 2.0 * Math.PI;
        const fan = P.fan * Math.pow(rpm / 3000.0, 2) * (Math.sin(S.fanph) * (1.0 + 0.3 * S.nz2) + 0.6 * S.nz2);
        this.b_mech[n_] = vm + mn + fan;
        this.b_pop0[n_] = pop0; this.b_pop1[n_] = pop1;
        this.rpmA[n_] = rpm;    // modulated (instantaneous) crank rpm -> aux (accessory kernel)
      }
      // ---------------- muffler FIR ----------------
      this._firBlock(m);
      // ---------------- pass 2 ----------------
      const M = this.M, pex = this.pex, peng = this.peng, pmas = this.pmas;
      const jt0 = this.jt_co[0], jt1 = this.jt_co[1], js0 = this.jt_st[0], js1 = this.jt_st[1];
      const r = P.radc;
      const bX0 = DL_X0 * DL_LEN, bX1 = DL_X1 * DL_LEN;
      for (let n_ = 0; n_ < m; n_++) {
        let ys0, ys1;
        {
          let v = this.b_v0[n_];
          if (V4) {
            const acc = v, lu = this.b_lu[n_];
            S.hflp0 += P.hfc * (acc - S.hflp0);
            if (lu !== 0.0) v = S.hflp0 + Math.pow(2.0, P.thf * lu) * (acc - S.hflp0);
            S.aflp0 += P.aflp * (this.b_pop0[n_] - S.aflp0);
            if (S.aflp0 !== 0.0) v += S.aflp0;
          }
          const yr = r * (S.rady0 + v - S.rad0); S.rad0 = v; S.rady0 = yr;
          let jn = this.b_jn0[n_]; for (let j = 0; j < jt0.length; j++) jn = bqr(jn, jt0[j], js0[j]);
          ys0 = yr + jn;
        }
        {
          let v = this.b_v1[n_];
          if (V4) {
            const acc = v, lu = this.b_lu[n_];
            S.hflp1 += P.hfc * (acc - S.hflp1);
            if (lu !== 0.0) v = S.hflp1 + Math.pow(2.0, P.thf * lu) * (acc - S.hflp1);
            S.aflp1 += P.aflp * (this.b_pop1[n_] - S.aflp1);
            if (S.aflp1 !== 0.0) v += S.aflp1;
          }
          const yr = r * (S.rady1 + v - S.rad1); S.rad1 = v; S.rady1 = yr;
          let jn = this.b_jn1[n_]; for (let j = 0; j < jt1.length; j++) jn = bqr(jn, jt1[j], js1[j]);
          ys1 = yr + jn;
        }
        let i0 = di[DL_X0]; dl[bX0 + i0] = ys0; i0 = (i0 + 1) & DL_MASK; di[DL_X0] = i0;
        let i1 = di[DL_X1]; dl[bX1 + i1] = ys1; i1 = (i1 + 1) & DL_MASK; di[DL_X1] = i1;
        const rd = (base, idx, d) => { const dI = Math.trunc(d), fr = d - dI, a = dl[base + ((idx - dI) & DL_MASK)], b = dl[base + ((idx - dI - 1) & DL_MASK)]; return a + fr * (b - a); };
        let eL = M[M_EXLL] * rd(bX0, i0, M[M_DLL] + 1.0) + M[M_EXRL] * rd(bX1, i1, M[M_DRL] + 1.0);
        let eR = M[M_EXLR] * rd(bX0, i0, M[M_DLR] + 1.0) + M[M_EXRR] * rd(bX1, i1, M[M_DRR] + 1.0);
        const gsum = 0.5 * (rd(bX0, i0, M[M_DLL] + M[M_GNDD] + 1.0) + rd(bX1, i1, M[M_DRR] + M[M_GNDD] + 1.0));
        eL += M[M_GND] * gsum; eR += M[M_GND] * gsum;
        for (let j = 0; j < pex.length; j++) { eL = bqr(eL, pex[j], this.pex_st[0][j]); eR = bqr(eR, pex[j], this.pex_st[1][j]); }
        let eng = M[M_HD] * this.b_hd[n_] + M[M_IN] * this.b_vi[n_] + M[M_MK] * this.b_mech[n_];
        for (let j = 0; j < peng.length; j++) eng = bqr(eng, peng[j], this.peng_st[j]);
        let L = eL + eng, R = eR + 0.95 * eng;
        for (let j = 0; j < pmas.length; j++) { L = bqr(L, pmas[j], this.pmas_st[0][j]); R = bqr(R, pmas[j], this.pmas_st[1][j]); }
        // ambience
        const mm = 0.5 * (L + R);
        let accr = 0.0;
        for (let c = 0; c < 4; c++) {
          const line = DL_C0 + c, base = line * DL_LEN, idx = di[line];
          const yv = dl[base + ((idx - rv_d[c]) & DL_MASK)];
          dl[base + idx] = mm + 0.62 * yv * 0.85 + 0.0; di[line] = (idx + 1) & DL_MASK;
          accr += yv;
        }
        let wl = accr, wr = accr;
        for (let c = 0; c < 2; c++) {
          let line = DL_A0 + c, base = line * DL_LEN, idx = di[line];
          let bv = dl[base + ((idx - rv_d[4 + c]) & DL_MASK)];
          dl[base + idx] = wl + 0.5 * bv; di[line] = (idx + 1) & DL_MASK; wl = bv - 0.5 * wl;
          line = DL_A2 + c; base = line * DL_LEN; idx = di[line];
          bv = dl[base + ((idx - rv_d[6 + c]) & DL_MASK)];
          dl[base + idx] = wr + 0.5 * bv; di[line] = (idx + 1) & DL_MASK; wr = bv - 0.5 * wr;
        }
        outL[off + n_] = L + M[M_REV] * wl;
        outR[off + n_] = R + M[M_REV] * wr;
      }
    }
  }
  // =====================================================================================
  // v4 kernel constants (engine_synth.py AFTERFIRE / AFTERFIRE_AMP, ir.py MUFFLER_IR / WET)
  // =====================================================================================
  const AFTERFIRE = { stock_reverse: [0.0, 1, 700.0, 1.0], stock_single: [0.01, 1, 800.0, 1.0], sport_turbo: [0.10, 1, 1800.0, 0.8],
    thrush: [0.18, 2, 2500.0, 0.6], glasspack: [0.25, 2, 3000.0, 0.6], cherry_bomb: [0.28, 3, 3200.0, 0.5], none: [0.35, 3, 6000.0, 0.4] };
  const AFTERFIRE_AMP = 0.4;
  const C_GAS = 530.0;
  const MUFFLER_IR = {
    stock_reverse: [[[C_GAS / 0.4, 6.0, 1.0], [1.84 * C_GAS / (Math.PI * 0.25), 5.0, 0.8], [550.0, 8.0, 0.6], [2 * C_GAS / 0.4, 3.0, 0.4]], [1.0, 2500.0, 0.2]],
    stock_single: [[[C_GAS / 0.4, 5.0, 1.0], [1.84 * C_GAS / (Math.PI * 0.23), 4.0, 0.7], [600.0, 7.0, 0.6]], [1.0, 2500.0, 0.2]],
    sport_turbo: [[[C_GAS / 0.9, 4.0, 0.8], [C_GAS / 0.45, 3.5, 1.0], [1.84 * C_GAS / (Math.PI * 0.2), 3.0, 0.6]], [1.5, 4000.0, 0.5]],
    glasspack: [[[C_GAS / 1.0, 2.5, 0.5], [C_GAS / 0.5, 2.0, 0.4]], [1.5, 5000.0, 1.0]],
    thrush: [[[C_GAS / 0.9, 2.5, 0.5], [C_GAS / 0.45, 2.0, 0.5]], [1.5, 4500.0, 1.0]],
    cherry_bomb: [[[C_GAS / 0.8, 2.5, 0.5], [C_GAS / 0.4, 2.0, 0.5]], [1.2, 5500.0, 1.0]],
    none: [[], [0.5, 8000.0, 0.0]],
  };
  const MUFFLER_WET = { stock_reverse: 0.25, stock_single: 0.25, sport_turbo: 0.25, glasspack: 0.3, thrush: 0.3, cherry_bomb: 0.3, none: 0.0 };
  const CABIN_WET = 0.30;
  // scipy butter(2, fc, 'lowpass') == RBJ low-pass with Q = 1/sqrt(2) (bilinear, prewarped at fc)
  const butterLp = (fc, fs) => bq('lp', fc, Math.SQRT1_2, fs);
  function sosRun(x, co, st) { for (let j = 0; j < co.length; j++) x = bqr(x, co[j], st[j]); return x; }

  /** ir.muffler_ir: procedural muffler-shell IR (modes + packing-noise burst), unit energy */
  function mufflerIr(kind, fs, n, seed) {
    n = n || 256; const rs = new NpRandomState(seed === undefined ? 7 : seed);
    const [modes, [nd, nlp, ng]] = MUFFLER_IR[kind], h = new Float64Array(n);
    for (const [f, dms, g] of modes) { const ph = rs.double() * 2 * Math.PI; for (let i = 0; i < n; i++) { const t = i / fs; h[i] += g * Math.exp(-t / (dms * 1e-3)) * Math.sin(2 * Math.PI * f * t + ph); } }
    if (ng > 0) {
      const nz = new Float64Array(n); for (let i = 0; i < n; i++) nz[i] = rs.randn() * Math.exp(-(i / fs) / (nd * 1e-3));
      const co = [butterLp(Math.min(nlp, 0.45 * fs), fs)], st = zst(co); let mx = 0;
      for (let i = 0; i < n; i++) { nz[i] = sosRun(nz[i], co, st); mx = Math.max(mx, Math.abs(nz[i])); }
      for (let i = 0; i < n; i++) h[i] += ng * nz[i] / (mx + 1e-12);
    }
    h[0] = 0; h[1] = 0; let e = 0; for (let i = 0; i < n; i++) e += h[i] * h[i];
    const s = Math.sqrt(e) + 1e-12; for (let i = 0; i < n; i++) h[i] /= s;
    return h;
  }
  /** ir.muffler_with_ir: h = h_TL + wet * norm(h_TL * ir), 768 taps */
  function mufflerWithIr(hTl, kind, fs, wet, ntap) {
    const [modes, nb] = MUFFLER_IR[kind];
    if (wet <= 0 || (!modes.length && nb[2] === 0)) return hTl;
    const ir = mufflerIr(kind, fs), w = new Float64Array(ntap);
    for (let n = 0; n < ntap; n++) { let a = 0; const lo = Math.max(0, n - ir.length + 1), hi = Math.min(n, hTl.length - 1); for (let i = lo; i <= hi; i++) a += hTl[i] * ir[n - i]; w[n] = a; }
    let eh = 0, ew = 0; for (let i = 0; i < hTl.length; i++) eh += hTl[i] * hTl[i]; for (let i = 0; i < ntap; i++) ew += w[i] * w[i];
    const k = wet * Math.sqrt(eh / (ew + 1e-12)), out = new Float64Array(ntap);
    for (let i = 0; i < ntap; i++) out[i] = (i < hTl.length ? hTl[i] : 0) + k * w[i];
    return out;
  }
  /** ir.cabin_ir: stereo cabin IR (modes of a 1.9 x 1.45 x 1.1 m interior + early reflections + 70 ms tail) */
  const cabinCache = {};
  function cabinIr(fs) {
    if (cabinCache[fs]) return cabinCache[fs];
    const dims = [1.9, 1.45, 1.1], rt60 = 0.07, secs = 0.18, seed = 11, n = Math.trunc(secs * fs), C_AIR_ = 343.0;
    const [L, W, H] = dims, modes = [];
    for (let nx = 0; nx < 3; nx++) for (let ny = 0; ny < 3; ny++) for (let nz = 0; nz < 2; nz++) {
      if (nx + ny + nz === 0) continue;
      const f = C_AIR_ / 2 * Math.sqrt((nx / L) ** 2 + (ny / W) ** 2 + (nz / H) ** 2);
      if (f < 350) modes.push([f, ((nx > 0) + (ny > 0) + (nz > 0)) === 1 ? 0.6 : 0.35]);
    }
    const out = [];
    for (let ch = 0; ch < 2; ch++) {
      const rs = new NpRandomState(seed + ch), h = new Float64Array(n);
      for (const [f, g] of modes) {
        const tau = 0.10 / 6.91 * (1.0 + 60.0 / f), ff = f * (1 + 0.01 * rs.randn()), ph = rs.double() * 6.28;
        for (let i = 0; i < n; i++) { const t = i / fs; h[i] += g * Math.exp(-t / tau) * Math.sin(2 * Math.PI * ff * t + ph); }
      }
      for (let i = 0; i < n; i++) h[i] *= 0.5;
      for (const [dms, g] of [[1.4, 0.7], [2.9, 0.5], [3.8, 0.45], [5.2, 0.3], [6.7, 0.2]]) {
        const i = Math.trunc((dms + 0.15 * rs.randn()) * 1e-3 * fs);
        if (0 < i && i < n) h[i] += g * (rs.double() < 0.7 ? 1 : -1);
      }
      const nz = new Float64Array(n); for (let i = 0; i < n; i++) nz[i] = rs.randn() * Math.exp(-(i / fs) * 6.91 / rt60);
      const co = [butterLp(4500, fs)], st = zst(co); let en = 0, eh = 0;
      for (let i = 0; i < n; i++) { nz[i] = sosRun(nz[i], co, st); en += nz[i] * nz[i]; }
      for (let i = 0; i < n; i++) eh += h[i] * h[i];
      const k = 0.6 / (Math.sqrt(en) + 1e-12) * Math.sqrt(eh + 1e-12);
      for (let i = 0; i < n; i++) h[i] += k * nz[i];
      for (let i = 0; i < Math.trunc(0.0008 * fs); i++) h[i] = 0;
      let e = 0; for (let i = 0; i < n; i++) e += h[i] * h[i];
      const s = Math.sqrt(e) + 1e-12; for (let i = 0; i < n; i++) h[i] /= s;
      out.push(h);
    }
    return (cabinCache[fs] = out);
  }
  /** stereo uniformly partitioned convolver (128-sample partitions, both channels in one complex FFT),
   *  y = x + wet * (x * ir): exact linear convolution, same as ir.BlockConvolver */
  class PartConv {
    constructor(irs, wet) {
      const B = CH, N = 2 * B, nt = irs[0].length, np = Math.ceil(nt / B);
      this.wet = wet; this.np = np; this.H0r = []; this.H0i = []; this.H1r = []; this.H1i = [];
      for (let p = 0; p < np; p++) {
        for (let c = 0; c < 2; c++) {
          const re = new Float64Array(N), im = new Float64Array(N);
          for (let t = 0; t < B && p * B + t < nt; t++) re[t] = irs[c][p * B + t];
          fft(re, im, -1);
          if (c === 0) { this.H0r.push(re); this.H0i.push(im); } else { this.H1r.push(re); this.H1i.push(im); }
        }
      }
      this.X0r = []; this.X0i = []; this.X1r = []; this.X1i = [];
      for (let p = 0; p < np; p++) { this.X0r.push(new Float64Array(N)); this.X0i.push(new Float64Array(N)); this.X1r.push(new Float64Array(N)); this.X1i.push(new Float64Array(N)); }
      this.head = 0; this.pL = new Float64Array(B); this.pR = new Float64Array(B);
      this.zr = new Float64Array(N); this.zi = new Float64Array(N); this.wr = new Float64Array(N); this.wi = new Float64Array(N);
      this.inL = new Float64Array(B); this.inR = new Float64Array(B); this.fill = 0; this.oL = new Float64Array(B); this.oR = new Float64Array(B);
    }
    clone() {
      const c = Object.create(PartConv.prototype);
      for (const k of Object.keys(this)) { const v = this[k]; c[k] = ArrayBuffer.isView(v) ? v.slice() : (Array.isArray(v) && v.length && ArrayBuffer.isView(v[0]) && /^X/.test(k)) ? v.map((a) => a.slice()) : v; }
      return c;
    }
    _block() {
      const B = CH, N = 2 * B, np = this.np, zr = this.zr, zi = this.zi;
      for (let i = 0; i < B; i++) { zr[i] = this.pL[i]; zi[i] = this.pR[i]; zr[B + i] = this.inL[i]; zi[B + i] = this.inR[i]; this.pL[i] = this.inL[i]; this.pR[i] = this.inR[i]; }
      fft(zr, zi, -1);
      const h = (this.head + np - 1) % np; this.head = h;
      const a0r = this.X0r[h], a0i = this.X0i[h], a1r = this.X1r[h], a1i = this.X1i[h];
      for (let k = 0; k < N; k++) {   // split the packed spectrum: X0 = (Z[k] + conj Z[N-k]) / 2, X1 = (Z[k] - conj Z[N-k]) / 2i
        const j = (N - k) & (N - 1), zr1 = zr[k], zi1 = zi[k], zr2 = zr[j], zi2 = -zi[j];
        a0r[k] = 0.5 * (zr1 + zr2); a0i[k] = 0.5 * (zi1 + zi2);
        a1r[k] = 0.5 * (zi1 - zi2); a1i[k] = -0.5 * (zr1 - zr2);
      }
      const wr = this.wr, wi = this.wi; wr.fill(0); wi.fill(0);
      for (let p = 0; p < np; p++) {
        const X = (h + p) % np, x0r = this.X0r[X], x0i = this.X0i[X], x1r = this.X1r[X], x1i = this.X1i[X];
        const h0r = this.H0r[p], h0i = this.H0i[p], h1r = this.H1r[p], h1i = this.H1i[p];
        for (let k = 0; k < N; k++) {
          const y0r = x0r[k] * h0r[k] - x0i[k] * h0i[k], y0i = x0r[k] * h0i[k] + x0i[k] * h0r[k];
          const y1r = x1r[k] * h1r[k] - x1i[k] * h1i[k], y1i = x1r[k] * h1i[k] + x1i[k] * h1r[k];
          wr[k] += y0r - y1i; wi[k] += y0i + y1r;     // pack Y0 + i Y1
        }
      }
      fft(wr, wi, +1);
      for (let i = 0; i < B; i++) { this.oL[i] = wr[B + i] / N; this.oR[i] = wi[B + i] / N; }
    }
    /** in place on L/R[off..off+n) ; n must keep 128-sample alignment over successive calls */
    process(L, R, off, n) {
      if (n % CH) throw new Error('PartConv: block alignment');
      for (let s = 0; s < n; s += CH) {
        for (let k = 0; k < CH; k++) { this.inL[k] = L[off + s + k]; this.inR[k] = R[off + s + k]; }
        this._block();
        for (let k = 0; k < CH; k++) { L[off + s + k] += this.wet * this.oL[k]; R[off + s + k] += this.wet * this.oR[k]; }
      }
    }
  }

  // =====================================================================================
  // startup.py: accessory kernel (_acc_kernel), one-shot transients, StartupSim controller
  // =====================================================================================
  const TWO_PI = 6.283185307179586;
  class XorShift64 {   // the kernels' _rand (xorshift64, 53-bit double)
    constructor(v) { this.hi = Math.floor(v / 4294967296) >>> 0; this.lo = v >>> 0; }
    rand() {
      let hi = this.hi, lo = this.lo, h, l;
      h = hi ^ ((hi << 13) | (lo >>> 19)); l = lo ^ (lo << 13); hi = h >>> 0; lo = l >>> 0;
      l = lo ^ ((lo >>> 7) | (hi << 25)); h = hi ^ (hi >>> 7); hi = h >>> 0; lo = l >>> 0;
      h = hi ^ ((hi << 17) | (lo >>> 15)); l = lo ^ (lo << 17); hi = h >>> 0; lo = l >>> 0;
      this.hi = hi; this.lo = lo;
      return ((hi >>> 11) * 4294967296 + (((hi << 21) | (lo >>> 11)) >>> 0)) * (1.0 / 9007199254740992.0);
    }
  }
  function svf(x, g, k, S, i) {   // TPT state-variable filter -> [lp, bp]
    const a1 = 1.0 / (1.0 + g * (g + k)), v3 = x - S[i + 1], v1 = a1 * S[i] + g * a1 * v3, v2 = S[i + 1] + g * v1;
    S[i] = 2.0 * v1 - S[i]; S[i + 1] = 2.0 * v2 - S[i + 1];
    SVF_LP = v2; return v1;
  }
  let SVF_LP = 0;
  const T_OM = 0, T_I = 1, T_V = 2, T_PHC = 3, T_PHP = 4, T_PHM = 5, T_PHCH = 6, T_PHCHF = 7, T_PHWP = 8, T_PHFP = 9, T_PHSQ = 10,
    T_LP1 = 11, T_LP2 = 12, T_LP3 = 13, T_BP1A = 14, T_BP2A = 16, T_BP3A = 18, T_FPA = 20, T_FPB = 21, T_FPC = 22, T_FPD = 23, T_SQA = 24, T_IAVG = 27, NS2 = 28;
  /** per-sample accessory kernel: starter motor + ring gear, charging unit, belt / water pump, friction, fuel pump, belt squeal */
  class AccKernel {
    constructor(Q, seed64, co) {
      this.Q = Q; this.C = { ston: 0, pin: 0, ich: 0, cut: 0, visc: 0, sq: 0, voc: 0 };
      this.S = new Float64Array(NS2); this.rng = new XorShift64(seed64); this.co = co; this.st = zst(co);
      const fs = Q.fs;
      this.a_up = 1.0 - Math.exp(-1.0 / (0.12 * fs)); this.a_dn = 1.0 - Math.exp(-1.0 / (0.55 * fs)); this.a_i = 1.0 - Math.exp(-1.0 / (0.004 * fs));
      this.gbp1 = Math.tan(Math.PI * 4200.0 / fs); this.gbp2 = Math.tan(Math.PI * 3000.0 / fs); this.glp1 = Math.tan(Math.PI * 1400.0 / fs);
      this.glp2 = Math.tan(Math.PI * 700.0 / fs); this.glp3 = Math.tan(Math.PI * 450.0 / fs); this.gsq = Math.tan(Math.PI * 2650.0 / fs);
    }
    clone(co) {
      const c = Object.create(AccKernel.prototype); Object.assign(c, this);
      c.Q = Object.assign({}, this.Q); c.C = Object.assign({}, this.C); c.S = this.S.slice();
      c.rng = Object.assign(Object.create(XorShift64.prototype), this.rng);
      c.co = co || this.co; c.st = co ? zst(co) : this.st.map((a) => a.slice());
      if (co) for (let j = 0; j < Math.min(co.length, this.st.length); j++) c.st[j] = this.st[j].slice();
      return c;
    }
    process(out, vout, rpm, n, off) {
      const Q = this.Q, C = this.C, S = this.S, fs = Q.fs, tw = 2.0 * Math.PI / fs, co = this.co, st = this.st;
      for (let i = 0; i < n; i++) {
        const cr = rpm[off + i];
        let V = S[T_V], itarget, ld;
        if (C.ston > 0.5) {
          const free = Q.free * V / 10.6, geared = cr * Q.ratio;
          if (C.pin > 0.5 && geared < free) S[T_OM] = geared; else S[T_OM] += (free - S[T_OM]) * this.a_up;
          ld = 1.0 - S[T_OM] / (free + 1e-6); if (ld < 0.0) ld = 0.0; if (ld > 1.0) ld = 1.0;
          itarget = Q.ifree + (Q.ilock - Q.ifree) * ld;
        } else { S[T_OM] -= S[T_OM] * this.a_dn; itarget = 0.0; ld = 0.0; }
        S[T_I] += (itarget - S[T_I]) * this.a_i;
        S[T_V] = C.voc - Q.rbatt * S[T_I];
        S[T_IAVG] += (S[T_I] - S[T_IAVG]) * 0.0005;
        if (vout) vout[i] = S[T_V];
        const om = S[T_OM], il = S[T_I] / Q.ilock;
        const nz1 = this.rng.rand() * 2.0 - 1.0, nz2 = this.rng.rand() * 2.0 - 1.0, nz3 = this.rng.rand() * 2.0 - 1.0;
        let y = 0.0;
        if (om > 1.0) {
          S[T_PHC] += tw * Q.segs * om / 60.0; S[T_PHP] += tw * Q.poles * om / 60.0;
          if (S[T_PHC] > TWO_PI) S[T_PHC] -= TWO_PI;
          if (S[T_PHP] > TWO_PI) S[T_PHP] -= TWO_PI;
          let spd = om / Q.free; if (spd > 1.2) spd = 1.2;
          const pc = S[T_PHC];
          const whine = Math.sin(pc) + 0.45 * Math.sin(2.0 * pc + 0.3) + 0.25 * Math.sin(3.0 * pc + 1.1);
          const buzz = Math.abs(Math.sin(S[T_PHP])) - 0.63;
          svf(nz1, this.glp1, 1.4, S, T_LP1); const lpw = SVF_LP;
          const bpb = svf(nz2, this.gbp1, 1.2, S, T_BP1A);
          y += Q.gst * ((0.25 + 0.75 * il) * Math.min(1.0, spd * 3.0) * 0.30 * whine * (1.0 + 0.5 * bpb)
            + 0.55 * il * buzz + 0.9 * spd * spd * lpw + 0.6 * Math.sqrt(il) * spd * bpb);
        }
        if (C.pin > 0.5 && cr > 1.0) {
          S[T_PHM] += tw * Q.ring * cr / 60.0; if (S[T_PHM] > TWO_PI) S[T_PHM] -= TWO_PI;
          const mesh = Math.sin(S[T_PHM]) * (0.4 + 1.2 * ld) + 0.8 * nz3 * (0.6 + 0.4 * Math.sin(S[T_PHM])) * (0.3 + ld);
          y += Q.gst * 0.35 * mesh * Math.min(1.0, cr / 60.0);
        }
        const wch = cr * Q.chr;
        if (wch > 1.0) {
          S[T_PHCH] += tw * Q.chord * wch / 60.0; S[T_PHCHF] += tw * Q.chfan * wch / 60.0;
          if (S[T_PHCH] > TWO_PI) S[T_PHCH] -= TWO_PI;
          if (S[T_PHCHF] > TWO_PI) S[T_PHCHF] -= TWO_PI;
          const ich = C.ich / Q.champs; let sp = wch / 3000.0; if (sp > 2.5) sp = 2.5;
          const bpc = svf(nz1, this.gbp2, 1.0, S, T_BP2A), ph = S[T_PHCH];
          if (Q.chtype < 0.5) {
            const tone = Math.sin(ph) + 0.2 * Math.sin(2.0 * ph);
            y += Q.gch * ((0.05 + 0.95 * ich) * Math.min(1.0, sp * 2.0) * 0.5 * tone + 0.25 * sp * sp * bpc * (1.0 + 0.4 * Math.sin(S[T_PHCHF])));
          } else {
            const tone = Math.sin(ph) + 0.6 * Math.sin(2.0 * ph + 0.5) + 0.35 * Math.sin(3.0 * ph), ld2 = C.cut * ich;
            y += Q.gch * ((0.08 + 0.9 * ld2) * Math.min(1.0, sp * 2.0) * 0.45 * tone * (1.0 + 0.8 * bpc)
              + (0.25 + 0.6 * ld2) * 0.6 * bpc * sp + 0.2 * sp * sp * bpc * (1.0 + 0.4 * Math.sin(S[T_PHCHF])));
          }
        }
        if (cr > 1.0) {
          const rr = cr / 1000.0;
          svf(nz2, this.glp2, 1.0, S, T_LP2); const lpb = SVF_LP;
          S[T_PHWP] += tw * Q.wpv * cr * Q.fanr / 60.0; if (S[T_PHWP] > TWO_PI) S[T_PHWP] -= TWO_PI;
          y += Q.gbelt * (0.6 * rr * lpb * (1.0 + 0.3 * Math.sin(S[T_PHWP])) + 0.08 * rr * Math.sin(S[T_PHWP]));
          svf(nz3, this.glp3, 1.0, S, T_LP3); const lpf = SVF_LP;
          y += Q.gfric * C.visc * rr * lpf;
          S[T_PHFP] += cr / 120.0 / fs;
          if (S[T_PHFP] >= 1.0) { S[T_PHFP] -= 1.0; S[T_FPA] = 0.7 + 0.6 * this.rng.rand(); S[T_FPB] = 0.0; }
        }
        // _fpump: 1.45 kHz tick (tau 3 ms) + 190 Hz thump (12 ms)
        let fp = 0.0;
        if (S[T_FPA] > 0.0) {
          S[T_FPB] += 1.0 / fs; const t = S[T_FPB];
          fp += S[T_FPA] * Math.exp(-t / 0.003) * Math.sin(2 * Math.PI * 1450.0 * t);
          fp += 0.8 * S[T_FPA] * Math.exp(-t / 0.012) * Math.sin(2 * Math.PI * 190.0 * t);
          if (t > 0.06) { S[T_FPA] = 0.0; S[T_FPB] = 0.0; }
        }
        y += Q.gfp * fp;
        if (C.sq > 0.0 && cr > 1.0) {
          S[T_PHSQ] += (110.0 + 0.05 * cr) / fs; let ex = 0.0;
          if (S[T_PHSQ] >= 1.0) { S[T_PHSQ] -= 1.0; ex = 1.0; }
          const sqb = svf(ex * 3.0 + 0.05 * nz1, this.gsq, 0.03, S, T_SQA);
          y += Q.gsq * C.sq * sqb;
        }
        for (let j = 0; j < co.length; j++) y = bqr(y, co[j], st[j]);
        out[off + i] = y;
      }
    }
  }
  function accQ(ss, fs, levels, cal, withStarter, gStart, starterBoost) {
    const Q = { fs, ratio: ss.ring_ratio, free: ss.st_free_rpm, ifree: ss.st_i_free, ilock: ss.st_i_lock, segs: ss.st_segs, poles: ss.st_poles,
      ring: ss.ring_teeth, rbatt: ss.r_batt, chr: ss.charge_ratio, chord: ss.charge_order, chfan: ss.charge_fan_blades,
      chtype: ss.charging === 'alternator' ? 0.0 : 1.0, champs: ss.charge_amps, fanr: ss.fan_ratio, wpv: ss.wp_vanes,
      gst: levels.GST, gch: levels.GCH, gbelt: levels.GBELT, gfp: levels.GFP, gfric: levels.GFRIC, gsq: levels.GSQ };
    const ga = cal.acc_gain === undefined ? 1.0 : cal.acc_gain;
    if (withStarter) { if (starterBoost) Q.gst *= 1.15; Q.gstBase = Q.gst; Q.gst *= gStart; } else { Q.gst = 0.0; Q.gstBase = 0.0; }
    Q.gch *= ga; Q.gbelt *= ga; Q.gfp *= ga; Q.gfric *= ga; Q.gsq *= ga;
    return Q;
  }
  /** startup.make_oneshot: modal click / clunk buffer (normalised), uses the controller's RandomState */
  function makeOneshot(spec, fs, rs) {
    const [bus, modes, [nd, nf, na]] = spec;
    let taumax = 0; for (const m of modes) taumax = Math.max(taumax, m[1]);
    const dur = Math.max(5 * taumax, nd) + 0.01, n = Math.trunc(dur * fs), y = new Float64Array(n);
    for (const [f0, tau, a] of modes) {
      const f = f0 * (1 + 0.03 * rs.randn()), ph = rs.double() * 6.28;
      // damped sinusoid by complex rotation (a*exp(-t/tau)*sin(2 pi f t + ph)), cheap enough for the audio thread
      const dec = Math.exp(-1 / (tau * fs)), w = 2 * Math.PI * f / fs, cr = dec * Math.cos(w), ci = dec * Math.sin(w);
      let zr = a * Math.cos(ph), zi = a * Math.sin(ph);
      for (let i = 0; i < n; i++) { y[i] += zi; const r = zr * cr - zi * ci; zi = zr * ci + zi * cr; zr = r; }
    }
    const co = bq('bp', nf, 1.2, fs), st = [0, 0], lim = Math.max(nd / 3, 1e-4);
    let mx = 0;
    for (let i = 0; i < n; i++) {
      const t = i / fs, g = rs.randn();
      const x = t < nd ? g * Math.exp(-t / lim) : 0.0;
      y[i] += na * bqr(x, co, st);
      y[i] *= Math.min(1.0, t / 0.0004);
      mx = Math.max(mx, Math.abs(y[i]));
    }
    for (let i = 0; i < n; i++) y[i] /= mx + 1e-9;
    return [bus, y];
  }

  /**
   * StartupSim (startup.py), block-based (256 samples) controller + the game hooks:
   *  - Python mode (validation): the scenario timeline runs as in render_startup().
   *  - game mode: key on/off come from ctl messages; after the catch and the choke flare, the crank
   *    speed is handed over to the game's physics rpm (the controller keeps writing the warm-up /
   *    choke / oil / charging state into the kernel).
   */
  const CB = 256;
  class StartupCtl {
    constructor(owner, sc) {
      const sp = owner.spec, D = owner.D, ss = D.START_SPECS[owner.carKey];
      this.sp = sp; this.ss = ss; this.sc = sc; this.fs = owner.fs; this.dt = CB / owner.fs;
      this.rs = new NpRandomState(sc.seed * 101 + (sc.cold ? 7 : 3));
      this.voc = sc.cold ? 12.35 : 12.65;
      this.t = 0.0; this.w = 0.0; this.wcrank = 0.0; this.revs = 0.0; this.phase = 'off'; this.ston = false; this.pin = false;
      this.fire_s = 0.0; this.nfire0 = 0; this.nevt0 = 0; this.t_catch = null; this.t_release = null; this.debt = 0.0; this.cut = false; this.oil = 0.0;
      const cold = sc.cold ? 1.0 : 0.0; this.cold0 = cold; this.choke0 = sc.cold ? 1.0 : 0.0; this.gas_warm = sc.cold ? 0.0 : 0.55;
      const r = this.rs;
      this.rev_prime = sc.cold ? (1.0 + 1.5 * r.double()) : (0.4 + 0.6 * r.double());
      this.rev_sputter = sc.cold ? (1.5 + 1.5 * r.double()) : (0.6 + 0.8 * r.double());
      this.react = 0.25 + 0.2 * r.double();
      this.runon_len = 1.6 + 1.4 * r.double();
      this.squeal = sc.belt_squeal < 0 ? (sc.cold ? 0.6 : 0.0) : sc.belt_squeal;
      this.active = [];        // [buffer, pos, bus]
      this.thr = 0; this.idle_now = sp.idle_rpm; this.firep = 0;
    }
    clone() { const c = Object.create(StartupCtl.prototype); Object.assign(c, this); c.rs = Object.assign(Object.create(NpRandomState.prototype), this.rs, { mt: this.rs.mt.slice() }); c.active = this.active.map((a) => a.slice()); return c; }
    fire(kind) { const [bus, buf] = makeOneshot(this.oneshots[kind], this.fs, this.rs); this.active.push([buf, 0, bus]); }
    choke() { if (!this.sc.cold || this.t_catch === null) return this.choke0; return Math.exp(-(this.t - this.t_catch) / this.sc.choke_time_s); }
    coldness() { if (this.t_catch === null) return this.cold0; return this.cold0 * Math.exp(-(this.t - this.t_catch) / (2.5 * this.sc.choke_time_s)); }
    runTarget() {
      const ss = this.ss, sp = this.sp, ch = this.choke();
      const idle = this.sc.cold ? sp.idle_rpm + (ss.fast_idle_rpm - sp.idle_rpm) * ch : sp.idle_rpm;
      if (this.t_catch === null) return [idle, idle];
      const tc = this.t - this.t_catch;
      let peak, tf, tr;
      if (this.sc.cold) { peak = ss.flare_cold_rpm; tf = 0.8; tr = 0.10; } else { peak = ss.flare_warm_rpm; tf = 0.5; tr = 0.08; }
      const g = Math.exp(-tc / tf) * (1.0 - Math.exp(-tc / tr));
      const gmax = Math.pow(tr / (tf + tr), tr / tf) * (tf / (tf + tr));
      return [idle + (peak - idle) * g / gmax, idle];
    }
    /** one 256-sample controller block: timeline, warm-up, crank speed, ignition, kernel P[], charging, accessory C */
    step(P, P0, acc, game) {
      const sc = this.sc, ss = this.ss, sp = this.sp, dt = this.dt, t = this.t;
      if (this.phase === 'off' && t >= sc.t_key_on) { this.phase = 'ign'; this.fire('key_on'); }
      if (this.phase === 'ign' && t >= sc.t_crank) {
        this.phase = 'crank'; this.fire('key_start'); if (ss.starter === 'ford_pe') this.fire('relay_on');
        this.t_sol = t + (ss.starter === 'ford_pe' ? 0.03 : 0.012); this.ston = false; this.pin = false;
      }
      if (this.phase === 'crank' && !this.ston && t >= this.t_sol) { this.fire('solenoid_in'); this.fire('pinion'); this.ston = this.pin = true; }
      if ((this.phase === 'crank' || this.phase === 'run') && this.t_catch !== null && this.t_release === null && t >= this.t_catch + this.react) {
        this.t_release = t; this.ston = this.pin = false; this.fire('solenoid_out'); this.fire('key_return'); if (ss.starter === 'ford_pe') this.fire('relay_off');
      }
      if ((this.phase === 'crank' || this.phase === 'run') && t >= sc.t_off) { this.phase = 'shut'; this.t_shut = t; this.fire('key_off'); this.ston = this.pin = false; }
      if (this.phase === 'ign' && t >= sc.t_off) { this.phase = 'stopped'; this.fire('key_off'); }   // game: key back before cranking
      const cold = this.coldness(), ch = this.choke();
      if (this.w > 60.0) { const tau_oil = sc.cold ? 2.6 : 0.6; this.oil += (1.0 - this.oil) * dt / tau_oil; }
      let heat = this.fire_s * (0.5 + 0.5 * Math.min(1.0, this.w / 1500.0));
      if (game && game.handed) heat = 1.0;     // game: a fully warmed engine converges to the fitted running model
      this.gas_warm += (heat - this.gas_warm) * dt / (sc.cold ? 6.0 : 3.0);
      const V = acc.S[T_V];
      if (this.ston && this.pin) {
        const cs = (sc.cold ? ss.crank_rpm_cold : ss.crank_rpm_warm) * (0.55 + 0.45 * V / 10.0) / 1.0;
        this.wcrank += (cs - this.wcrank) * Math.min(1.0, dt / 0.09);
      } else this.wcrank = 0.0;
      const [tgt, idle_now] = this.runTarget(); this.idle_now = idle_now;
      let fired_driven = Math.pow(this.fire_s, 1.5) * tgt;
      if (this.phase === 'shut') {
        if (sc.runon && t < this.t_shut + this.runon_len) { const runon_t = 330.0 + 120.0 * Math.sin(7.0 * t) * this.rs.double(); fired_driven = Math.pow(this.fire_s, 0.7) * runon_t; }
        else fired_driven = 0.0;
      }
      if (fired_driven > this.w) this.w += (fired_driven - this.w) * Math.min(1.0, dt / 0.08);
      else if (this.phase === 'shut' || this.fire_s < 0.3) { const drag = (220.0 + 1.1 * this.w) * (1.0 + 0.6 * cold); this.w = Math.max(fired_driven, this.w - drag * dt); }
      else this.w += (fired_driven - this.w) * Math.min(1.0, dt / 0.5);
      if (this.ston && this.pin) this.w = Math.max(this.w, this.wcrank);
      if (game && this.phase === 'run' && game.blend > 0) this.w += (game.rpm - this.w) * game.blend;   // game hand-over
      if (game && this.phase === 'shut' && game.inGear && game.rpm > this.w) this.w = game.rpm;          // rolling in gear
      if (this.phase === 'stopped') this.w = 0.0;
      if (this.phase === 'shut' && this.w < 45.0 && this.w > 0.0) { this.w = 0.0; this.phase = 'stopped'; this.t_stop = t; this.fire('stop_thunk'); }
      if (this.phase === 'crank' && this.t_catch === null && this.w > 1.8 * Math.max(this.wcrank, 100.0) && this.fire_s > 0.4) { this.t_catch = t; this.phase = 'run'; }
      this.revs += this.w * dt / 60.0;
      let firep;
      if (this.phase === 'off' || this.phase === 'ign' || this.phase === 'stopped') firep = 0.0;
      else if (this.phase === 'crank') {
        const cr = this.revs;
        if (cr < this.rev_prime) firep = 0.0;
        else if (cr < this.rev_prime + this.rev_sputter) { const u = (cr - this.rev_prime) / this.rev_sputter; firep = sc.cold ? (0.15 + 0.35 * u) : (0.4 + 0.5 * u); }
        else firep = sc.cold ? 0.9 : 0.98;
      } else if (this.phase === 'run') firep = sc.cold ? (0.93 + 0.065 * (1.0 - cold)) : 0.997;
      else firep = (sc.runon && t < this.t_shut + this.runon_len) ? 0.25 + 0.3 * this.rs.double() : 0.0;
      this.firep = firep;
      P.firep = firep; P.pumpa = 0.085 * P0.amp; P.crankmod = 0.20 * Math.min(1.0, Math.max(0.0, (480.0 - this.w) / 300.0));
      P.idle = Math.max(idle_now, 1.0); P.rich = P0.rich + 0.6 * ch; P.misf = P0.misf + 0.04 * cold + 0.05 * ch;
      P.sig0 = P0.sig0 + 0.06 * cold; P.idlemap = P0.idlemap + 0.08 * ch;
      const hyd = sp.lifters !== 'solid', noil = 1.0 - this.oil;
      P.tick = P0.tick * (hyd ? (1.0 + 5.0 * Math.pow(noil, 1.5)) : (1.0 + 0.8 * noil + 0.35 * cold));
      P.mech = P0.mech * (1.0 + 1.0 * cold);
      P.temp0 = sc.ambient_k + (P0.temp0 - sc.ambient_k) * this.gas_warm;
      P.tempa = P0.tempa * this.fire_s;
      P.hx = Math.min(0.9, P0.hx + ss.heat_riser * cold);
      let thr = 0.0;
      if (!sc.cold && this.phase === 'crank') thr = 0.3;
      if (!sc.cold && this.phase === 'run' && t < this.t_catch + this.react + 0.15) thr = 0.3;
      if (this.t_catch !== null && this.phase === 'run') thr = Math.max(thr, 0.16 * Math.exp(-(t - this.t_catch) / 1.2) * (sc.cold ? 1.0 : 0.5));
      this.thr = thr;
      if (this.ston) this.debt += acc.S[T_I] * dt;
      const wch = this.w * ss.charge_ratio; let avail;
      if (ss.charging === 'generator') {
        if (!this.cut && this.w >= ss.gen_cutin_rpm) { this.cut = true; this.fire('cutout'); }
        else if (this.cut && this.w < ss.gen_cutout_rpm) { this.cut = false; this.fire('cutout'); }
        avail = this.cut ? ss.charge_amps * Math.min(1.0, Math.max(0.0, (wch - 1350.0) / 1400.0)) : 0.0;
      } else avail = (this.phase === 'crank' || this.phase === 'run') ? ss.charge_amps * Math.min(1.0, Math.max(0.0, (wch - 900.0) / 2600.0)) : 0.0;
      const demand = (this.phase === 'run' || this.phase === 'crank') ? 6.0 + this.debt / 25.0 : 0.0;
      const ich = Math.min(avail, demand);
      this.debt = Math.max(0.0, this.debt - Math.max(0.0, ich - 6.0) * dt);
      const C = acc.C;
      C.ston = this.ston ? 1.0 : 0.0; C.pin = this.pin ? 1.0 : 0.0; C.ich = ich; C.cut = (this.cut || ss.charging === 'alternator') ? 1.0 : 0.0;
      C.visc = 1.0 + 1.5 * cold;
      let sq = 0.0;
      if (this.squeal > 0 && this.t_catch !== null) { const tc = t - this.t_catch; if (0.1 < tc && tc < 0.9) sq = this.squeal * Math.sin(Math.PI * (tc - 0.1) / 0.8); }
      C.sq = sq; C.voc = this.voc;
    }
    /** after the block: fire-fraction bookkeeping from the kernel counters, advance the clock */
    after(S) {
      const nf = S.nfire, ne = S.nevt, dnf = nf - this.nfire0, dne = ne - this.nevt0;
      this.nfire0 = nf; this.nevt0 = ne;
      if (dne > 0) { const a = 1.0 - Math.pow(1.0 - 0.22, dne); this.fire_s += a * (dnf / dne - this.fire_s); }
      else if (this.phase === 'shut' || this.phase === 'stopped') this.fire_s *= 0.9;
      this.t += this.dt;
    }
  }

  // =====================================================================================
  // car_audio.py: CarbHiss, GearWhine, CarSound (+ the start-up sequence for the game)
  // =====================================================================================
  function butterTable(D, fs, name) {
    const tab = D.BUTTER, k = String(fs);
    let row = tab[k];
    if (!row) { let best = null, bd = 1e9; for (const kk of Object.keys(tab)) { const d = Math.abs(+kk - fs); if (d < bd) { bd = d; best = kk; } } row = tab[best]; }
    return row[name].map((s) => [s[0], s[1], s[2], s[4], s[5]]);   // sos [b0 b1 b2 a0 a1 a2] -> [b0 b1 b2 a1 a2]
  }
  class CarbHiss {
    constructor(fs, D, seed) {
      seed = seed === undefined ? 9 : seed; this.fs = fs; this.rs = new NpRandomState(seed); this.rs_m = new NpRandomState(seed + 1000);
      this.co = butterTable(D, fs, 'hiss'); this.st = zst(this.co); this.ph = 0.0;
    }
    clone() { const c = Object.create(CarbHiss.prototype); Object.assign(c, this); c.rs = cloneRs(this.rs); c.rs_m = cloneRs(this.rs_m); c.st = this.st.map((a) => a.slice()); return c; }
    /** adds g_in * HISS_K * hiss to out (L += x, R += 0.97 x is done by the caller) */
    process(out, off, n, vac, rpm, red, k) {
      const flow = Math.min(1.5, rpm / red), g = 0.05 * Math.pow(Math.max(0.0, vac - 0.4), 1.5) * Math.pow(flow, 0.8);
      if (!(g > 0)) return;
      const f = 5200.0 + 1800.0 * flow, w = 2 * Math.PI * f / this.fs, ph0 = this.ph;
      for (let i = 0; i < n; i++) out[off + i] = sosRun(this.rs.randn(), this.co, this.st);
      for (let i = 0; i < n; i++) out[off + i] = k * g * (out[off + i] + 0.15 * Math.sin(ph0 + w * (i + 1)) * (1 + 0.5 * this.rs_m.randn()));
      this.ph = (ph0 + w * n) % (2 * Math.PI);
    }
  }
  class GearWhine {
    constructor(fs, d) { this.fs = fs; this.d = d || { mesh_teeth: 0, whine: 0.0, smooth: 1.0 }; this.ph = 0.0; }
    clone() { return Object.assign(Object.create(GearWhine.prototype), this); }
    process(out, off, n, rpm, gear, tq, k) {
      if (!gear || !this.d.mesh_teeth) return;
      const f = rpm / 60.0 * this.d.mesh_teeth, w = 2 * Math.PI * f / this.fs, ph0 = this.ph;
      const g = this.d.whine * (0.25 + Math.pow(Math.abs(tq), 0.6)) * Math.min(1.5, rpm / 3000.0);
      for (let i = 0; i < n; i++) { const t = ph0 + w * (i + 1); out[off + i] += k * g * (Math.sin(t) + 0.3 * Math.sin(2 * t + 0.7)); }
      this.ph = (ph0 + w * n) % (2 * Math.PI);
    }
  }
  function cloneRs(rs) { return Object.assign(Object.create(NpRandomState.prototype), rs, { mt: rs.mt.slice() }); }

  const POSN = (lis) => lis.label || lis.position;
  /**
   * CarSoundV34: what the game hears from model v3 / v4 at one listener position.
   * Same interface as the v2 Synth for ListenerSource (render, cloneWithListener, spec), plus
   * ctl(m) (key on/off, gear, torque), selfGated (it fades itself in/out), report().
   *  opts: ver (3|4), mode: 'game' | 'startup' (validation: Python StartupSim) | 'run' (validation: CarSound),
   *        sc (start scenario), dry (no CarSound extras / gain, = StartupSim output), noAcc.
   */
  class CarSoundV34 {
    constructor(spec, lis, fs, opts) {
      opts = opts || {};
      this.opts = opts; this.ver = opts.ver || 4; this.fs = fs; this.D = spec.v34; this.carKey = spec.car_key;
      if (!this.D || !this.carKey) throw new Error('v34: spec needs car_key and v34 data');
      this.syn = new EngineSynthV34(spec, lis, fs, { ver: this.ver, fir: opts.fir || 'fft' });
      this.spec = this.syn.spec; this.P0 = Object.assign({}, this.syn.P);
      this.selfGated = opts.mode === 'game'; this.listenerAware = true;
      this.mode = opts.mode || 'game';
      this.state = this.mode === 'startup' ? 'startup' : (this.mode === 'run' ? 'running' : 'off');
      this.key = false; this.warm = false; this.starts = 0; this.game = { rpm: 0, thr: 0, gear: 0, tq: 0, handed: false, blend: 0, inGear: false };
      this.extra = new Float64Array(256); this.osb = new Float64Array(CB); this.osc = new Float64Array(CB); this.cpos = 0;
      this.tmpL = new Float64Array(256); this.tmpR = new Float64Array(256);
      this.hiss = new CarbHiss(fs, this.D); this.whine = new GearWhine(fs, this.D.DRIVETRAIN[this.carKey]);
      this.ctlr = null; this.acc = null; this.fadeG = this.mode === 'game' ? 0 : 1; this.silentT = 0;
      this._position(lis, null);
      if (this.mode === 'startup') this._beginStartup(Object.assign({ cold: true, runon: false, t_key_on: 0.6, t_crank: 1.5, t_off: 20.0, choke_time_s: 14.0, ambient_k: 288.0, belt_squeal: -1.0, seed: 1 }, opts.sc || {}));
      else if (this.mode === 'run') this._beginRunning();
    }
    /** per-position quantities (startup.py POS_ACC / cal, car_audio mix gain, cabin IR) */
    _position(lis, prev) {
      const D = this.D, pos = this.syn.listener.position, cal = D.START_CAL[this.carKey] || {};
      const M = this.syn.M;
      this.g_in = M[M_IN]; this.g_mk = M[M_MK];
      const pa = D.POS_ACC[pos] || [1.0, 0.0];
      this.g_bay = pa[0] * M[M_MK]; this.g_cab = pa[1];
      this.g_start = cal[pos] === undefined ? 1.0 : cal[pos];
      this.os_co = this.syn.peng.map((c) => c.slice()); this.os_st = prev ? prev.os_st.map((a) => a.slice()) : zst(this.os_co);
      this.cab_co = chain(pos.indexOf('in_cabin') === 0 ? [['hp', 300, 0.7]] : [['lp', 2500, 0.7], ['hp', 300, 0.7]], this.fs);
      this.cab_st = zst(this.cab_co);
      if (prev) for (let j = 0; j < Math.min(this.cab_st.length, prev.cab_st.length); j++) this.cab_st[j] = prev.cab_st[j].slice();
      const cab = pos.indexOf('in_cabin') === 0;
      this.conv = (this.ver >= 4 && cab && !this.opts.dry) ? (prev && prev.conv ? prev.conv.clone() : new PartConv(cabinIr(this.fs), CABIN_WET)) : null;
      const gdb = this.ver >= 4 && !this.opts.dry ? (((D.MIX_GAINS[this.carKey] || {})[POSN(this.syn.listener)]) || 0.0) : 0.0;
      this.g = Math.pow(10, gdb / 20);
      if (this.acc) {
        if (this.acc.Q.gstBase !== undefined) this.acc.Q.gst = this.acc.Q.gstBase * (this.ctlr ? this.g_start : 0);
        this.acc.co = this.syn.peng; this.acc.st = this.acc.st.map((a) => a.slice());
      }
    }
    cloneWithListener(lis) {
      const c = Object.create(CarSoundV34.prototype); Object.assign(c, this);
      c.syn = this.syn.cloneWithListener(lis); c.spec = c.syn.spec;
      c.game = Object.assign({}, this.game); c.extra = this.extra.slice(); c.osb = this.osb.slice(); c.osc = this.osc.slice();
      c.tmpL = new Float64Array(256); c.tmpR = new Float64Array(256);
      c.hiss = this.hiss.clone(); c.whine = this.whine.clone();
      c.ctlr = this.ctlr ? this.ctlr.clone() : null;
      c.acc = this.acc ? this.acc.clone(c.syn.peng) : null;
      c._position(lis, this);
      return c;
    }
    _beginStartup(sc) {
      const D = this.D, ss = D.START_SPECS[this.carKey], cal = D.START_CAL[this.carKey] || {};
      // fresh engine (StartupSim builds a new synth): same listener
      const lis = this.syn.listener;
      this.syn = new EngineSynthV34(this.spec, lis, this.fs, { ver: this.ver, fir: this.opts.fir || 'fft' });
      this.P0 = Object.assign({}, this.syn.P);
      this.ctlr = new StartupCtl(this, sc); this.ctlr.oneshots = D.ONESHOTS;
      const Q = accQ(ss, this.fs, D.LEVELS, cal, true, this.g_start, ss.starter === 'ford_pe');
      this.acc = new AccKernel(Q, 0xA5A5A5A5 + 977 * sc.seed, this.syn.peng);
      this.acc.S[T_V] = this.ctlr.voc; this.acc.C.voc = this.ctlr.voc;
      this.syn.S.tex = sc.ambient_k + (sc.cold ? 0.0 : 200.0);
      this.syn._rpm = 0.0; this.cpos = 0; this.osb.fill(0); this.osc.fill(0);
      this.game.handed = false; this.game.blend = 0; this.state = 'startup';
      this._position(lis, null);
    }
    /** car_audio.CarSound with startup.AccessoryLayer (engine already running, warm) */
    _beginRunning() {
      const D = this.D, ss = D.START_SPECS[this.carKey], cal = D.START_CAL[this.carKey] || {};
      this.ctlr = null;
      const Q = accQ(ss, this.fs, D.LEVELS, cal, false, 1, false);
      this.acc = this.opts.noAcc ? null : new AccKernel(Q, 0xC0FFEE + 31 * this.spec.seed, this.syn.peng);
      if (this.acc) { const C = this.acc.C; C.ich = 8.0; C.visc = 1.0; C.voc = 12.65; this.acc.S[T_V] = 12.65; }
      this.cut = false; this.state = 'running';
    }
    /** key-off while running without a start-up controller (model switched while running): a warm
     *  controller in the run phase takes over, so the shutdown (coast-down, stop thunk) is the same */
    _adoptController() {
      const sc = { cold: false, runon: false, t_key_on: -1, t_crank: -1, t_off: Infinity, choke_time_s: 14.0, ambient_k: 288.0, belt_squeal: 0, seed: 1 + this.starts++ };
      const c = new StartupCtl(this, sc); c.oneshots = this.D.ONESHOTS;
      c.phase = 'run'; c.t_catch = -100; c.t_release = -100; c.fire_s = 1.0; c.oil = 1.0; c.gas_warm = 1.0; c.w = Math.max(this.game.rpm, 1);
      c.nfire0 = this.syn.S.nfire; c.nevt0 = this.syn.S.nevt; c.revs = 100;
      this.ctlr = c; this.cpos = 0; this.game.handed = true; this.game.bt = 1; this.game.blend = 1; this.state = 'startup';
      if (!this.acc) { const ss = this.D.START_SPECS[this.carKey]; this.acc = new AccKernel(accQ(ss, this.fs, this.D.LEVELS, this.D.START_CAL[this.carKey] || {}, false, 1, false), 0xC0FFEE + 31 * this.spec.seed, this.syn.peng); this.acc.S[T_V] = 12.65; }
    }
    ctl(m) {
      const g = this.game;
      if (m.gear !== undefined) g.gear = m.gear;
      if (m.tq !== undefined) g.tq = m.tq;
      if (m.warm !== undefined) this.warm = !!m.warm;
      if (this.mode !== 'game') return;
      if (m.running !== undefined || m.starter !== undefined) {
        const on = !!(m.running || m.starter);
        if (on && !this.key) {
          this.key = true;
          if (this.state === 'off' || (this.ctlr && (this.ctlr.phase === 'stopped' || this.ctlr.phase === 'shut'))) {
            this._beginStartup({ cold: !this.warm, runon: false, t_key_on: 0.0, t_crank: 0.3, t_off: Infinity, choke_time_s: 14.0, ambient_k: 288.0, belt_squeal: -1.0, seed: 1 + this.starts++ });
            this.fadeG = 1;
          } else if (this.state === 'running' && !this.ctlr) { /* already running (model switched) */ }
        } else if (!on && this.key) {
          this.key = false;
          if (!this.ctlr && this.state === 'running') this._adoptController();
          if (this.ctlr && this.ctlr.sc.t_off === Infinity) { this.ctlr.sc = Object.assign({}, this.ctlr.sc, { t_off: this.ctlr.t }); }
        }
      }
    }
    /** the engine was already running when this source was created (model / car switch) */
    initRunning(on) { if (this.mode !== 'game') return; if (on) { this.key = true; this._beginRunning(); this.fadeG = 1; } }
    report() {
      const c = this.ctlr;
      if (this.state === 'off') return { phase: 'off', rpm: 0 };
      if (!c) return { phase: this.state === 'running' ? 'running' : 'off', rpm: this.game.rpm };
      return { phase: c.phase, rpm: c.w, idle: c.idle_now, handed: this.game.handed, cold: c.sc.cold };
    }
    render(n, rpm, thr, outL, outR, off) {
      off = off || 0;
      if (this.state === 'off') { for (let i = 0; i < n; i++) { outL[off + i] = 0; outR[off + i] = 0; } return; }
      for (let s = 0; s < n;) {
        const m = this.ctlr ? Math.min(n - s, CB - this.cpos) : Math.min(n - s, 256);
        this._part(m, rpm, thr, outL, outR, off + s);
        s += m;
      }
    }
    _part(m, rpmIn, thrIn, outL, outR, off) {
      const syn = this.syn, P = syn.P, c = this.ctlr, g = this.game;
      g.rpm = rpmIn; g.thr = thrIn;
      let rpm = rpmIn, thr = thrIn;
      if (c) {
        if (this.cpos === 0) {
          if (this.mode === 'game') {
            // hand-over to the game's physics rpm: after the flare (or as soon as the driver touches the gas)
            if (c.phase === 'run') {
              const tc = c.t - c.t_catch, hold = c.sc.cold ? 1.6 : 1.0;
              if (!g.handed && (tc > hold || (tc > c.react + 0.1 && thrIn > 0.05))) { g.handed = true; g.bt = 0; }
              if (g.handed) { g.bt += c.dt; g.blend = Math.min(1, g.bt / 0.35); } else g.blend = 0;
            } else g.blend = 0;
            g.inGear = !!g.gear;
          }
          c.step(P, this.P0, this.acc, this.mode === 'game' ? g : null);
          if (this.mode === 'game' && c.phase === 'run' && g.handed && g.blend >= 1) c.w = rpmIn;
          // one-shot transients for this controller block (startup.py mixes them per 256-sample block)
          const bay = this.osb, cab = this.osc; bay.fill(0); cab.fill(0); const keep = []; let any = false;
          for (const a of c.active) {
            const buf = a[0], p = a[1], mm = Math.min(CB, buf.length - p), dst = a[2] === 'bay' ? bay : cab;
            for (let i = 0; i < mm; i++) dst[i] += buf[p + i];
            if (mm > 0) any = true;
            a[1] = p + mm; if (a[1] < buf.length) keep.push(a);
          }
          c.active = keep;
          if (any) {
            for (let i = 0; i < CB; i++) {
              bay[i] = sosRun(bay[i], this.os_co, this.os_st);
              cab[i] = sosRun(cab[i], this.cab_co, this.cab_st);
            }
          }
          const k = this.D.LEVELS.ONESHOT;
          for (let i = 0; i < CB; i++) bay[i] = k * (this.g_bay * this.g_start * bay[i] + this.g_cab * cab[i]);
          this.r0 = this.cpos === 0 ? syn._rpm : this.r0; this.t0 = syn._thr; this.wT = c.w; this.thT = (this.mode === 'game' && g.handed) ? thrIn : c.thr;
          if (this.mode === 'game' && c.phase === 'run' && g.handed) this.thT = c.thr + (thrIn - c.thr) * g.blend;
        }
        // ramp the controller block's targets over its 256 samples, rendered in parts
        const u = (this.cpos + m) / CB;
        rpm = this.r0 + (this.wT - this.r0) * u; thr = this.t0 + (this.thT - this.t0) * u;
      }
      // fluid coupling smooths the cycle-to-cycle roughness in gear (CarSound, v4)
      let sig0Saved = null;
      if (this.ver >= 4 && !this.opts.dry) { sig0Saved = P.sig0; if (g.gear) P.sig0 = P.sig0 * ((this.D.DRIVETRAIN[this.carKey] || {}).smooth || 1.0); }
      const L = this.tmpL, R = this.tmpR;
      syn.render(m, rpm, thr, L, R, 0);
      if (sig0Saved !== null) P.sig0 = sig0Saved;
      // v4 CarSound extras: carb hiss past the closed plate + gearbox mesh whine (estimated vacuum, as car_audio.py)
      if (this.ver >= 4 && !this.opts.dry) {
        const ex = this.extra; for (let i = 0; i < m; i++) ex[i] = 0;
        const r = Math.max(rpm, 1.0), vac = 1.0 - Math.min(1.0, Math.max(0.18, (0.35 * 600 + 9000.0 * Math.pow(Math.max(thr, 0), 1.4)) / r));
        this.hiss.process(ex, 0, m, vac, rpm, this.spec.redline, this.g_in * this.D.HISS_K);
        this.whine.process(ex, 0, m, rpm, g.gear, g.tq, this.g_mk);
        for (let i = 0; i < m; i++) { L[i] += ex[i]; R[i] += 0.97 * ex[i]; }
      }
      // accessories (startup kernel during the start-up, AccessoryLayer while running)
      if (this.acc) {
        const ab = this.extra;
        if (!c) {
          let mean = 0; for (let i = 0; i < m; i++) mean += syn.aux[i]; mean /= m;
          const ss = this.D.START_SPECS[this.carKey];
          if (ss.charging === 'generator') { if (!this.cut && mean >= ss.gen_cutin_rpm) this.cut = true; else if (this.cut && mean < ss.gen_cutout_rpm) this.cut = false; this.acc.C.cut = this.cut ? 1.0 : 0.0; }
          else this.acc.C.cut = 1.0;
        }
        this.acc.process(ab, null, syn.aux, m, 0);
        const ga = c ? this.g_bay : this.g_mk * ((this.D.POS_ACC[syn.listener.position] || [1.0])[0]);
        for (let i = 0; i < m; i++) { L[i] += ga * ab[i]; R[i] += 0.95 * ga * ab[i]; }
      }
      if (c) {
        for (let i = 0; i < m; i++) { const v = this.osb[this.cpos + i]; L[i] += v; R[i] += v; }
        this.cpos += m;
        if (this.cpos >= CB) { this.cpos = 0; c.after(syn.S); this.r0 = syn._rpm; }
      }
      if (this.conv) this.conv.process(L, R, 0, m);
      let gg = this.g;
      if (this.mode === 'game') {
        // self-gating: silent before key-on, fade out after the stop thunk / a key-off without a controller
        const done = c && c.phase === 'stopped' && c.active.length === 0 && (c.t - (c.t_stop || c.t)) > 0.4;
        const ignOff = c && c.phase === 'stopped' && !c.t_stop && c.active.length === 0;
        if (done || ignOff || this.state === 'fadeout') {
          this.fadeG = Math.max(0, this.fadeG - m / (0.05 * this.fs));
          if (this.fadeG <= 0) { this.state = 'off'; this.ctlr = null; this.key = false; }
        }
      }
      for (let i = 0; i < m; i++) { const k = gg * (this.mode === 'game' ? this.fadeG : 1); outL[off + i] = k * L[i]; outR[off + i] = k * R[i]; }
    }
  }

  const MODEL = (ver) => ({ id: 'v' + ver, Synth: CarSoundV34, synthOpts: { ver, mode: 'game' }, startup: true, positions: ['rear', 'rear_far', 'engine_bay', 'in_cabin'],
    EngineSynth: EngineSynthV34, CarSound: CarSoundV34, StartupCtl, AccKernel, mufflerFir, mufflerWithIr, mufflerIr, cabinIr, PartConv, positionMatrix, fft });
  return [MODEL(3), MODEL(4)];
},
function engineModuleBrm9(core) {
    // model parameters only (tools/export_brm9_params.py from engine-sound brm9.py + fitted_params_v9_*.json): merged PHYS / GEOM
    // per fitted set + the kernel constants brm9.BRM9Synth derives from them (q); COMMON = module constants of brm9.py / brm6.py
    var COMMON = {"stub_z":[0.4601578514017764,0.3596506232513735,0.09710951191066246,-0.07351765578769823,-0.07233413778823487,0.5381187567527944,-1.6807549872233718,0.09340929735988178,0.7033247016822283,-1.008014400498587,-1.050341006284517,0.4752506204861838,-0.0916655439252554,0.7426428208118669,2.3733095299198372,-1.8663459820689405],"mays_rpm":[0.0,2000.0,3500.0,5000.0,6000.0,7000.0,8000.0,9000.0,10000.0,11000.0,12000.0,12600.0],"mays_bhp":[0.0,18.0,50.0,100.0,175.0,250.0,335.0,412.0,525.0,585.0,600.0,560.0],"fire_order":[1,10,6,13,2,16,5,11,8,15,3,12,7,9,4,14],"j_crank":0.15,"persp":{"rear":{"ears":[[-5.4,0.09,1.0],[-5.4,-0.09,1.0]],"ground":0.75,"shadow_far_db":-3.0,"wind":0.0,"outside":true,"ambience":0.0},"side":{"ears":[[0.09,4.7,1.2],[-0.09,4.7,1.2]],"ground":0.75,"shadow_far_db":-9.0,"wind":0.0,"outside":true,"ambience":0.02},"onboard":{"ears":[[-1.2,0.09,0.75],[-1.2,-0.09,0.75]],"ground":0.0,"shadow_far_db":-2.0,"wind":0.02,"outside":false,"ambience":0.0}}};
    var SETS = {"v9_1":{"phys":{"stub_len":0.5414880233462647,"stub_sig":0.04095965620559825,"k_evo":4.5,"k_mot":1.1,"t_fire":1250.0,"t_mot":520.0,"t_stub0":471.1377477209218,"t_stuba":350.0,"k_nl":0.8951929515775512,"mod_max":0.45,"ro_loss":0.97,"weak":0.891530446025328,"weak_cyl":1,"misfire":0.1252235566149115,"cov":0.05143494614644671,"cov_idle":0.005181153021192375,"misfire_idle":0.20581154220665474,"exit_pitch":0.055899608452020355,"tors_deg":0.4859439587753051,"k_jet":0.024388675867880218,"jet_x":4.0,"whine":0.5382794983771708,"whine_w":0.26970031294761004,"whine_lp":6000.0,"blade":0.002,"rush":0.002,"wind":0.2026334683806248,"prim_len":0.35,"prim_d":0.06,"sec_f":0.55,"sec_r":0.25,"tail_len":1.0,"a2r":2.0,"atr":2.5,"t_pipe":750.0,"stub_c":0.0,"chg_c":0.18707186416778254,"wind_fc":405.40482361018735,"pad_lp":1509.4569422360382,"out_lp":3155.113511539341,"out_mix":0.6591947256480176,"shadow_db":-9.0,"onb_lp":20000.0,"onb_mix":0.0,"mech":0.0,"fuel_model":0.0,"fuel_meoh":0.8,"fuel_petrol":0.1,"fuel_acetone":0.1,"lam":0.8,"stub_dir":0.0,"stub_yaw":0.0,"stub_pitch":0.0,"onb_mic":0.0,"onb_aim":0.0,"onb_el":0.0,"onb_hp":0.0,"gnd_sigma":0.0,"da_sigma":0.0,"far_sigma":0.0,"half_deg":0.0,"stub_track":0.0,"da_lp":0.0,"out_hp":0.0,"da_az":0.0,"far_az":0.0,"tg_x":-2.0,"tg_y":0.0,"tg_z":0.95,"far_dl":0.0,"bank_dx":0.0,"onb_ground":0.0,"tun_w":4.5,"tun_h":5.5,"tun_cw":0.0,"tun_cc":0.0,"chg_lin":0.0,"onb_x":-1.2,"onb_y":0.0,"onb_z":0.75,"pad_sep":0.0,"gnd_fc":0.0,"sh_fc":0.0,"sh_db":-9.0,"idle_map":0.3,"ov_map":0.15,"mech_mod":0.6,"ab_on":0.0,"ab_xlim":0.5,"ab_xw":0.25,"ab_eff":0.3,"ab_tau":0.0015,"ab_mouth":0.5,"fl_on":0.0,"fl_eff":0.2,"fl_turb":0.5,"fl_fc":500.0},"geom":{"bore":0.04953,"stroke":0.04826,"cr":7.8,"conrod":0.096,"dv":0.0215,"lmax":0.0062,"evo":102.58855726702774,"dur":300.0,"camx":1.453387839370317,"cd":0.65,"stub_d":0.034,"drive_ratio":4.15,"blower_lobes":2,"blade_z":[12,12],"pr_max":5.373551634521821,"idle_rpm":2500.0,"redline":12000.0,"stub_y":0.45,"stub_z":0.25,"tail_x":-1.25,"tail_y":0.45,"tail_z":0.15,"b_mirror":1},"q":{"RG":290.0,"GAM":1.32,"TFIRE":1250.0,"KEVO":4.5,"TSTUB0":471.1377477209218,"TSTUBA":350.0,"ABON":0.0,"XLIM":0.0,"XW":0.0,"ABEFF":0.0,"ABTAU":0.0,"ABMOUTH":0.0,"AFRF":0.0,"LHVE":0.0,"TCH":0.0,"FLON":0.0,"FLK":0.0,"FLTURB":0.0}},"v9_2":{"phys":{"stub_len":0.55,"stub_sig":0.04349732181520666,"k_evo":4.5,"k_mot":1.1,"t_fire":1250.0,"t_mot":520.0,"t_stub0":497.34445043449153,"t_stuba":350.0,"k_nl":1.5,"mod_max":0.45,"ro_loss":0.97,"weak":0.8243873591057389,"weak_cyl":1,"misfire":0.13460058068130187,"cov":0.017984912644741766,"cov_idle":0.0818881435523372,"misfire_idle":0.22775540189090837,"exit_pitch":0.065,"tors_deg":0.0,"k_jet":0.02,"jet_x":4.0,"whine":0.5382794983771708,"whine_w":0.26970031294761004,"whine_lp":6000.0,"blade":0.002,"rush":0.02,"wind":0.23897842502259198,"prim_len":0.35,"prim_d":0.06,"sec_f":0.55,"sec_r":0.25,"tail_len":1.0,"a2r":2.0,"atr":2.5,"t_pipe":750.0,"stub_c":0.0,"chg_c":-0.2,"wind_fc":514.2647991368509,"pad_lp":1107.911500472259,"out_lp":2395.5068175045653,"out_mix":1.0,"shadow_db":-9.0,"onb_lp":4100.120168882837,"onb_mix":0.8496904268672767,"mech":0.010328302997279824,"fuel_model":0.0,"fuel_meoh":0.8,"fuel_petrol":0.1,"fuel_acetone":0.1,"lam":0.8,"stub_dir":0.0,"stub_yaw":0.0,"stub_pitch":0.0,"onb_mic":0.0,"onb_aim":0.0,"onb_el":0.0,"onb_hp":0.0,"gnd_sigma":0.0,"da_sigma":0.0,"far_sigma":0.0,"half_deg":0.0,"stub_track":0.0,"da_lp":0.0,"out_hp":0.0,"da_az":0.0,"far_az":0.0,"tg_x":-2.0,"tg_y":0.0,"tg_z":0.95,"far_dl":0.0,"bank_dx":0.0,"onb_ground":0.0,"tun_w":4.5,"tun_h":5.5,"tun_cw":0.0,"tun_cc":0.0,"chg_lin":0.0,"onb_x":-1.2,"onb_y":0.0,"onb_z":0.75,"pad_sep":1.0,"gnd_fc":0.0,"sh_fc":997.3098617859288,"sh_db":-8.470900265610908,"idle_map":0.21668471290867022,"ov_map":0.04093851072666962,"mech_mod":0.6,"ab_on":0.0,"ab_xlim":0.5,"ab_xw":0.25,"ab_eff":0.3,"ab_tau":0.0015,"ab_mouth":0.5,"fl_on":0.0,"fl_eff":0.2,"fl_turb":0.5,"fl_fc":500.0},"geom":{"bore":0.04953,"stroke":0.04826,"cr":7.8,"conrod":0.096,"dv":0.0215,"lmax":0.0062,"evo":102.01096601086984,"dur":300.0,"camx":1.4767886436763162,"cd":0.65,"stub_d":0.034,"drive_ratio":4.15,"blower_lobes":2,"blade_z":[12,12],"pr_max":5.696214039276613,"idle_rpm":2500.0,"redline":12000.0,"stub_y":0.45,"stub_z":0.25,"tail_x":-1.25,"tail_y":0.45,"tail_z":0.15,"b_mirror":1},"q":{"RG":290.0,"GAM":1.32,"TFIRE":1250.0,"KEVO":4.5,"TSTUB0":497.34445043449153,"TSTUBA":350.0,"ABON":0.0,"XLIM":0.0,"XW":0.0,"ABEFF":0.0,"ABTAU":0.0,"ABMOUTH":0.0,"AFRF":0.0,"LHVE":0.0,"TCH":0.0,"FLON":0.0,"FLK":0.0,"FLTURB":0.0}},"v9_3":{"phys":{"stub_len":0.5425000000000001,"stub_sig":0.04349732181520666,"k_evo":4.5,"k_mot":1.1,"t_fire":1250.0,"t_mot":520.0,"t_stub0":497.34445043449153,"t_stuba":350.0,"k_nl":1.5,"mod_max":0.45,"ro_loss":0.97,"weak":0.8243873591057389,"weak_cyl":1,"misfire":0.13460058068130187,"cov":0.017984912644741766,"cov_idle":0.0818881435523372,"misfire_idle":0.22775540189090837,"exit_pitch":0.075,"tors_deg":0.0,"k_jet":0.02,"jet_x":4.0,"whine":0.5382794983771708,"whine_w":0.26970031294761004,"whine_lp":6000.0,"blade":0.002,"rush":0.02,"wind":0.258978425022592,"prim_len":0.35,"prim_d":0.06,"sec_f":0.55,"sec_r":0.25,"tail_len":1.0,"a2r":2.0,"atr":2.5,"t_pipe":750.0,"stub_c":0.0,"chg_c":-0.2,"wind_fc":514.2647991368509,"pad_lp":1107.911500472259,"out_lp":2395.5068175045653,"out_mix":1.0,"shadow_db":-9.0,"onb_lp":3500.120168882837,"onb_mix":0.9996904268672767,"mech":0.010328302997279824,"fuel_model":0.0,"fuel_meoh":0.8,"fuel_petrol":0.1,"fuel_acetone":0.1,"lam":0.8,"stub_dir":0.0,"stub_yaw":0.0,"stub_pitch":0.0,"onb_mic":0.0,"onb_aim":0.0,"onb_el":0.0,"onb_hp":0.0,"gnd_sigma":0.0,"da_sigma":0.0,"far_sigma":0.0,"half_deg":0.0,"stub_track":0.0,"da_lp":0.0,"out_hp":0.0,"da_az":0.0,"far_az":0.0,"tg_x":-2.0,"tg_y":0.0,"tg_z":0.95,"far_dl":0.0,"bank_dx":0.0,"onb_ground":0.0,"tun_w":4.5,"tun_h":5.5,"tun_cw":0.0,"tun_cc":0.0,"chg_lin":0.02,"onb_x":-1.125,"onb_y":0.2,"onb_z":0.88,"pad_sep":1.0,"gnd_fc":0.0,"sh_fc":1297.3098617859287,"sh_db":-3,"idle_map":0.21668471290867022,"ov_map":0.04093851072666962,"mech_mod":0.6,"ab_on":0.0,"ab_xlim":0.5,"ab_xw":0.25,"ab_eff":0.3,"ab_tau":0.0015,"ab_mouth":0.5,"fl_on":0.0,"fl_eff":0.2,"fl_turb":0.5,"fl_fc":500.0},"geom":{"bore":0.04953,"stroke":0.04826,"cr":7.8,"conrod":0.096,"dv":0.0215,"lmax":0.0062,"evo":102.01096601086984,"dur":300.0,"camx":1.4767886436763162,"cd":0.65,"stub_d":0.034,"drive_ratio":4.15,"blower_lobes":2,"blade_z":[12,12],"pr_max":5.696214039276613,"idle_rpm":2500.0,"redline":12000.0,"stub_y":0.45,"stub_z":0.25,"tail_x":-1.25,"tail_y":0.45,"tail_z":0.15,"b_mirror":1},"q":{"RG":290.0,"GAM":1.32,"TFIRE":1250.0,"KEVO":4.5,"TSTUB0":497.34445043449153,"TSTUBA":350.0,"ABON":0.0,"XLIM":0.0,"XW":0.0,"ABEFF":0.0,"ABTAU":0.0,"ABMOUTH":0.0,"AFRF":0.0,"LHVE":0.0,"TCH":0.0,"FLON":0.0,"FLK":0.0,"FLTURB":0.0}},"v9_4":{"phys":{"stub_len":0.5425000000000001,"stub_sig":0.04349732181520666,"k_evo":4.5,"k_mot":1.1,"t_fire":1250.0,"t_mot":520.0,"t_stub0":497.34445043449153,"t_stuba":350.0,"k_nl":1.5,"mod_max":0.45,"ro_loss":0.97,"weak":0.8243873591057389,"weak_cyl":1,"misfire":0.13460058068130187,"cov":0.017984912644741766,"cov_idle":0.0818881435523372,"misfire_idle":0.22775540189090837,"exit_pitch":0.075,"tors_deg":0.0,"k_jet":0.02,"jet_x":4.0,"whine":0.5382794983771708,"whine_w":0.26970031294761004,"whine_lp":6000.0,"blade":0.002,"rush":0.02,"wind":0.258978425022592,"prim_len":0.35,"prim_d":0.06,"sec_f":0.55,"sec_r":0.25,"tail_len":1.0,"a2r":2.0,"atr":2.5,"t_pipe":750.0,"stub_c":0.04,"chg_c":-0.2,"wind_fc":514.2647991368509,"pad_lp":1107.911500472259,"out_lp":2395.5068175045653,"out_mix":1.0,"shadow_db":-9.0,"onb_lp":3500.120168882837,"onb_mix":0.9996904268672767,"mech":0.010328302997279824,"fuel_model":0.0,"fuel_meoh":0.8,"fuel_petrol":0.1,"fuel_acetone":0.1,"lam":0.8,"stub_dir":0.0,"stub_yaw":0.0,"stub_pitch":0.0,"onb_mic":0.0,"onb_aim":0.0,"onb_el":0.0,"onb_hp":0.0,"gnd_sigma":0.0,"da_sigma":0.0,"far_sigma":0.0,"half_deg":0.0,"stub_track":0.0,"da_lp":0.0,"out_hp":0.0,"da_az":0.0,"far_az":0.0,"tg_x":-2.0,"tg_y":0.0,"tg_z":0.95,"far_dl":0.0,"bank_dx":-0.03,"onb_ground":0.0,"tun_w":4.5,"tun_h":5.5,"tun_cw":0.0,"tun_cc":0.0,"chg_lin":0.02,"onb_x":-1.075,"onb_y":0.2,"onb_z":0.88,"pad_sep":1.0,"gnd_fc":0.0,"sh_fc":1297.3098617859287,"sh_db":-3,"idle_map":0.21668471290867022,"ov_map":0.04093851072666962,"mech_mod":0.6,"ab_on":0.0,"ab_xlim":0.5,"ab_xw":0.25,"ab_eff":0.3,"ab_tau":0.0015,"ab_mouth":0.5,"fl_on":0.0,"fl_eff":0.2,"fl_turb":0.5,"fl_fc":500.0},"geom":{"bore":0.04953,"stroke":0.04826,"cr":7.8,"conrod":0.096,"dv":0.0215,"lmax":0.0062,"evo":102.01096601086984,"dur":300.0,"camx":1.4767886436763162,"cd":0.65,"stub_d":0.034,"drive_ratio":4.15,"blower_lobes":2,"blade_z":[12,12],"pr_max":5.696214039276613,"idle_rpm":2500.0,"redline":12000.0,"stub_y":0.45,"stub_z":0.25,"tail_x":-1.25,"tail_y":0.45,"tail_z":0.15,"b_mirror":1},"q":{"RG":290.0,"GAM":1.32,"TFIRE":1250.0,"KEVO":4.5,"TSTUB0":497.34445043449153,"TSTUBA":350.0,"ABON":0.0,"XLIM":0.0,"XW":0.0,"ABEFF":0.0,"ABTAU":0.0,"ABMOUTH":0.0,"AFRF":0.0,"LHVE":0.0,"TCH":0.0,"FLON":0.0,"FLK":0.0,"FLTURB":0.0}},"v9_6":{"phys":{"stub_len":0.5425000000000001,"stub_sig":0.04349732181520666,"k_evo":4.5,"k_mot":1.1,"t_fire":1250.0,"t_mot":520.0,"t_stub0":497.34445043449153,"t_stuba":350.0,"k_nl":1.5,"mod_max":0.45,"ro_loss":0.97,"weak":0.8243873591057389,"weak_cyl":1,"misfire":0.13460058068130187,"cov":0.017984912644741766,"cov_idle":0.0818881435523372,"misfire_idle":0.22775540189090837,"exit_pitch":0.075,"tors_deg":0.0,"k_jet":0.02,"jet_x":4.0,"whine":0.5382794983771708,"whine_w":0.26970031294761004,"whine_lp":6000.0,"blade":0.002,"rush":0.02,"wind":0.258978425022592,"prim_len":0.35,"prim_d":0.06,"sec_f":0.55,"sec_r":0.25,"tail_len":1.0,"a2r":2.0,"atr":2.5,"t_pipe":750.0,"stub_c":0.04,"chg_c":-0.2,"wind_fc":514.2647991368509,"pad_lp":1107.911500472259,"out_lp":2395.5068175045653,"out_mix":1.0,"shadow_db":-9.0,"onb_lp":3500.120168882837,"onb_mix":0.9996904268672767,"mech":0.010328302997279824,"fuel_model":0.0,"fuel_meoh":0.8,"fuel_petrol":0.1,"fuel_acetone":0.1,"lam":0.8,"stub_dir":0.0,"stub_yaw":0.0,"stub_pitch":0.0,"onb_mic":0.0,"onb_aim":0.0,"onb_el":0.0,"onb_hp":0.0,"gnd_sigma":0.0,"da_sigma":0.0,"far_sigma":0.0,"half_deg":0.0,"stub_track":0.0,"da_lp":0.0,"out_hp":30.0,"da_az":3.0,"far_az":12.0,"tg_x":-2.0,"tg_y":0.0,"tg_z":0.95,"far_dl":0.0,"bank_dx":-0.03,"onb_ground":0.0,"tun_w":4.5,"tun_h":5.5,"tun_cw":0.0,"tun_cc":0.0,"chg_lin":0.02,"onb_x":-1.075,"onb_y":0.2,"onb_z":0.88,"pad_sep":1.0,"gnd_fc":0.0,"sh_fc":1500,"sh_db":-3,"idle_map":0.21668471290867022,"ov_map":0.04093851072666962,"mech_mod":0.6,"ab_on":0.0,"ab_xlim":0.5,"ab_xw":0.25,"ab_eff":0.3,"ab_tau":0.0015,"ab_mouth":0.5,"fl_on":0.0,"fl_eff":0.2,"fl_turb":0.5,"fl_fc":500.0},"geom":{"bore":0.04953,"stroke":0.04826,"cr":7.8,"conrod":0.096,"dv":0.0215,"lmax":0.0062,"evo":102.01096601086984,"dur":300.0,"camx":1.4767886436763162,"cd":0.65,"stub_d":0.034,"drive_ratio":4.15,"blower_lobes":2,"blade_z":[12,12],"pr_max":5.696214039276613,"idle_rpm":2500.0,"redline":12000.0,"stub_y":0.45,"stub_z":0.25,"tail_x":-1.25,"tail_y":0.45,"tail_z":0.15,"b_mirror":1},"q":{"RG":290.0,"GAM":1.32,"TFIRE":1250.0,"KEVO":4.5,"TSTUB0":497.34445043449153,"TSTUBA":350.0,"ABON":0.0,"XLIM":0.0,"XW":0.0,"ABEFF":0.0,"ABTAU":0.0,"ABMOUTH":0.0,"AFRF":0.0,"LHVE":0.0,"TCH":0.0,"FLON":0.0,"FLK":0.0,"FLTURB":0.0}},"v9_8":{"phys":{"stub_len":0.5425000000000001,"stub_sig":0.04349732181520666,"k_evo":4.5,"k_mot":1.1,"t_fire":1050.0,"t_mot":520.0,"t_stub0":445.89313136443366,"t_stuba":313.7917711582222,"k_nl":1.7,"mod_max":0.45,"ro_loss":0.97,"weak":0.5993873591057388,"weak_cyl":1,"misfire":0.13460058068130187,"cov":0.07798491264474176,"cov_idle":0.0818881435523372,"misfire_idle":0.22775540189090837,"exit_pitch":0.06999999999999999,"tors_deg":0.0,"k_jet":0.02,"jet_x":4.0,"whine":0.5382794983771708,"whine_w":0.26970031294761004,"whine_lp":6000.0,"blade":0.002,"rush":0.02,"wind":0.258978425022592,"prim_len":0.35,"prim_d":0.06,"sec_f":0.55,"sec_r":0.25,"tail_len":1.0,"a2r":2.0,"atr":2.5,"t_pipe":750.0,"stub_c":-0.0049999999999999975,"chg_c":-0.2,"wind_fc":514.2647991368509,"pad_lp":1107.911500472259,"out_lp":2395.5068175045653,"out_mix":0.85,"shadow_db":-9.0,"onb_lp":3500.120168882837,"onb_mix":0.9996904268672767,"mech":0.010328302997279824,"fuel_model":1.0,"fuel_meoh":0.8,"fuel_petrol":0.1,"fuel_acetone":0.1,"lam":0.8,"stub_dir":0.0,"stub_yaw":0.0,"stub_pitch":0.0,"onb_mic":1.0,"onb_aim":0.0,"onb_el":30.0,"onb_hp":0.0,"gnd_sigma":0.0,"da_sigma":0.0,"far_sigma":30000000.0,"half_deg":0.0,"stub_track":0.0,"da_lp":0.0,"out_hp":30.0,"da_az":3.0,"far_az":12.0,"tg_x":-2.0,"tg_y":0.0,"tg_z":0.95,"far_dl":0.0,"bank_dx":-0.0375,"onb_ground":0.0,"tun_w":4.5,"tun_h":5.5,"tun_cw":0.0,"tun_cc":0.0,"chg_lin":0.17,"onb_x":-1.075,"onb_y":0.2,"onb_z":0.88,"pad_sep":1.0,"gnd_fc":0.0,"sh_fc":1500,"sh_db":-3,"idle_map":0.21668471290867022,"ov_map":0.04093851072666962,"mech_mod":0.6,"ab_on":0.0,"ab_xlim":0.5,"ab_xw":0.25,"ab_eff":0.3,"ab_tau":0.0015,"ab_mouth":0.5,"fl_on":0.0,"fl_eff":0.2,"fl_turb":0.5,"fl_fc":500.0},"geom":{"bore":0.04953,"stroke":0.04826,"cr":7.8,"conrod":0.096,"dv":0.0215,"lmax":0.0062,"evo":102.01096601086984,"dur":290.0,"camx":1.4767886436763162,"cd":0.65,"stub_d":0.034,"drive_ratio":4.15,"blower_lobes":2,"blade_z":[12,12],"pr_max":5.696214039276613,"idle_rpm":2500.0,"redline":12000.0,"stub_y":0.45,"stub_z":0.25,"tail_x":-1.25,"tail_y":0.45,"tail_z":0.15,"b_mirror":1},"q":{"RG":312.6611364809158,"GAM":1.2939061828269274,"TFIRE":1108.1818469895627,"KEVO":4.3381427487802275,"TSTUB0":470.60064178609315,"TSTUBA":331.1793757449951,"ABON":0.0,"XLIM":0.0,"XW":0.0,"ABEFF":0.0,"ABTAU":0.0,"ABMOUTH":0.0,"AFRF":0.0,"LHVE":0.0,"TCH":0.0,"FLON":0.0,"FLK":0.0,"FLTURB":0.0}},"v9_9":{"phys":{"stub_len":0.5425000000000001,"stub_sig":0.04349732181520666,"k_evo":5.5,"k_mot":1.1,"t_fire":1050.0,"t_mot":520.0,"t_stub0":445.89313136443366,"t_stuba":313.7917711582222,"k_nl":1.7,"mod_max":0.45,"ro_loss":0.97,"weak":0.5993873591057388,"weak_cyl":1,"misfire":0.0,"cov":0.07798491264474176,"cov_idle":0.0818881435523372,"misfire_idle":0.0,"exit_pitch":0.06999999999999999,"tors_deg":0.0,"k_jet":0.02,"jet_x":4.0,"whine":0.5382794983771708,"whine_w":0.26970031294761004,"whine_lp":6000.0,"blade":0.002,"rush":0.02,"wind":0.188978425022592,"prim_len":0.35,"prim_d":0.06,"sec_f":0.55,"sec_r":0.25,"tail_len":1.0,"a2r":2.0,"atr":2.5,"t_pipe":750.0,"stub_c":-0.0049999999999999975,"chg_c":-0.05000000000000002,"wind_fc":514.2647991368509,"pad_lp":1107.911500472259,"out_lp":2395.5068175045653,"out_mix":0.85,"shadow_db":-9.0,"onb_lp":3500.120168882837,"onb_mix":0.8996904268672767,"mech":0.010328302997279824,"fuel_model":1.0,"fuel_meoh":0.8,"fuel_petrol":0.1,"fuel_acetone":0.1,"lam":0.8,"stub_dir":0.0,"stub_yaw":0.0,"stub_pitch":0.0,"onb_mic":1.0,"onb_aim":0.0,"onb_el":30.0,"onb_hp":0.0,"gnd_sigma":0.0,"da_sigma":0.0,"far_sigma":30000000.0,"half_deg":0.0,"stub_track":0.0,"da_lp":0.0,"out_hp":30.0,"da_az":-3.0,"far_az":12.0,"tg_x":-2.0,"tg_y":0.0,"tg_z":0.95,"far_dl":0.0,"bank_dx":-0.0375,"onb_ground":0.0,"tun_w":4.5,"tun_h":5.5,"tun_cw":0.3,"tun_cc":0.0,"chg_lin":0.17,"onb_x":-1.075,"onb_y":0.2,"onb_z":0.88,"pad_sep":1.0,"gnd_fc":0.0,"sh_fc":1500,"sh_db":-3,"idle_map":0.21668471290867022,"ov_map":0.05093851072666962,"mech_mod":0.6,"ab_on":1.0,"ab_xlim":0.5,"ab_xw":0.25,"ab_eff":0.675,"ab_tau":0.0015,"ab_mouth":0.5,"fl_on":1.0,"fl_eff":0.2,"fl_turb":0.5,"fl_fc":250.0},"geom":{"bore":0.04953,"stroke":0.04826,"cr":7.8,"conrod":0.096,"dv":0.0215,"lmax":0.0062,"evo":102.01096601086984,"dur":290.0,"camx":1.4767886436763162,"cd":0.65,"stub_d":0.034,"drive_ratio":4.15,"blower_lobes":2,"blade_z":[12,12],"pr_max":5.696214039276613,"idle_rpm":2500.0,"redline":12000.0,"stub_y":0.45,"stub_z":0.25,"tail_x":-1.25,"tail_y":0.45,"tail_z":0.15,"b_mirror":1},"q":{"RG":312.6611364809158,"GAM":1.2939061828269274,"TFIRE":1108.1818469895627,"KEVO":5.30217447073139,"TSTUB0":470.60064178609315,"TSTUBA":331.1793757449951,"ABON":1.0,"XLIM":0.5,"XW":0.25,"ABEFF":0.675,"ABTAU":0.0015,"ABMOUTH":0.5,"AFRF":6.023670229007634,"LHVE":18384203.562340967,"TCH":445.6077429377809,"FLON":1.0,"FLK":0.29338643748870946,"FLTURB":0.5}},"v9_10":{"phys":{"stub_len":0.5425000000000001,"stub_sig":0.04349732181520666,"k_evo":5.5,"k_mot":1.1,"t_fire":1050.0,"t_mot":520.0,"t_stub0":445.89313136443366,"t_stuba":313.7917711582222,"k_nl":1.7,"mod_max":0.45,"ro_loss":0.97,"weak":0.5993873591057388,"weak_cyl":1,"misfire":0.0,"cov":0.07798491264474176,"cov_idle":0.0818881435523372,"misfire_idle":0.0,"exit_pitch":0.06999999999999999,"tors_deg":0.0,"k_jet":0.02,"jet_x":4.0,"whine":0.638,"whine_w":0.18,"whine_lp":6000.0,"blade":0.002,"rush":0.02,"whine_res":0.5,"blade_res":0.0,"rush_res":0.25,"whine_ov":4.0,"surge_q":0.2,"wind":0.188978425022592,"prim_len":0.35,"prim_d":0.06,"sec_f":0.55,"sec_r":0.25,"tail_len":1.0,"a2r":2.0,"atr":2.5,"t_pipe":750.0,"stub_c":-0.0049999999999999975,"chg_c":-0.05000000000000002,"wind_fc":514.2647991368509,"pad_lp":1107.911500472259,"out_lp":2395.5068175045653,"out_mix":0.85,"shadow_db":-9.0,"onb_lp":3500.120168882837,"onb_mix":0.8996904268672767,"mech":0.010328302997279824,"fuel_model":1.0,"fuel_meoh":0.8,"fuel_petrol":0.1,"fuel_acetone":0.1,"lam":0.8,"stub_dir":0.0,"stub_yaw":0.0,"stub_pitch":0.0,"onb_mic":1.0,"onb_aim":0.0,"onb_el":30.0,"onb_hp":0.0,"gnd_sigma":0.0,"da_sigma":0.0,"far_sigma":30000000.0,"half_deg":0.0,"stub_track":0.0,"da_lp":0.0,"out_hp":30.0,"da_az":-3.0,"far_az":12.0,"tg_x":-2.0,"tg_y":0.0,"tg_z":0.95,"far_dl":0.0,"bank_dx":-0.0375,"onb_ground":0.0,"tun_w":4.5,"tun_h":5.5,"tun_cw":0.3,"tun_cc":0.0,"chg_lin":0.17,"onb_x":-1.075,"onb_y":0.2,"onb_z":0.88,"pad_sep":1.0,"gnd_fc":0.0,"sh_fc":1500,"sh_db":-3,"idle_map":0.21668471290867022,"ov_map":0.05093851072666962,"mech_mod":0.6,"ab_on":1.0,"ab_xlim":0.5,"ab_xw":0.25,"ab_eff":0.675,"ab_tau":0.0015,"ab_mouth":0.5,"fl_on":1.0,"fl_eff":0.2,"fl_turb":0.5,"fl_fc":250.0},"geom":{"bore":0.04953,"stroke":0.04826,"cr":7.8,"conrod":0.096,"dv":0.0215,"lmax":0.0062,"evo":102.01096601086984,"dur":290.0,"camx":1.4767886436763162,"cd":0.65,"stub_d":0.034,"drive_ratio":4.15,"blower_lobes":2,"blade_z":[12,12],"pr_max":5.696214039276613,"idle_rpm":2500.0,"redline":12000.0,"stub_y":0.45,"stub_z":0.25,"tail_x":-1.25,"tail_y":0.45,"tail_z":0.15,"b_mirror":1},"q":{"RG":312.6611364809158,"GAM":1.2939061828269274,"TFIRE":1108.1818469895627,"KEVO":5.30217447073139,"TSTUB0":470.60064178609315,"TSTUBA":331.1793757449951,"ABON":1.0,"XLIM":0.5,"XW":0.25,"ABEFF":0.675,"ABTAU":0.0015,"ABMOUTH":0.5,"AFRF":6.023670229007634,"LHVE":18384203.562340967,"TCH":445.6077429377809,"FLON":1.0,"FLK":0.29338643748870946,"FLTURB":0.5,"WRES":0.5,"BRES":0.0,"RRES":0.25,"WOV":4.0,"SURGQ":0.2}},"v9_11":{"phys":{"stub_len":0.5425000000000001,"stub_sig":0.04349732181520666,"k_evo":5.5,"k_mot":1.1,"t_fire":1050.0,"t_mot":520.0,"t_stub0":445.89313136443366,"t_stuba":313.7917711582222,"k_nl":1.7,"mod_max":0.45,"ro_loss":0.97,"weak":0.5993873591057388,"weak_cyl":1,"misfire":0.0,"cov":0.07798491264474176,"cov_idle":0.0818881435523372,"misfire_idle":0.0,"exit_pitch":0.06999999999999999,"tors_deg":0.0,"k_jet":0.02,"jet_x":4.0,"whine":0.538,"whine_w":0.18,"whine_lp":6000.0,"blade":0.002,"rush":0.02,"whine_res":0.35,"blade_res":0.0,"rush_res":0.25,"whine_ov":2.5,"surge_q":0.2,"wind":0.1,"prim_len":0.35,"prim_d":0.06,"sec_f":0.55,"sec_r":0.25,"tail_len":1.0,"a2r":2.0,"atr":2.5,"t_pipe":750.0,"stub_c":-0.0049999999999999975,"chg_c":-0.05000000000000002,"wind_fc":439.0,"pad_lp":1107.911500472259,"out_lp":2395.5068175045653,"out_mix":0.85,"shadow_db":-9.0,"onb_lp":4400.0,"onb_mix":0.8996904268672767,"mech":0.0503,"fuel_model":1.0,"fuel_meoh":0.8,"fuel_petrol":0.1,"fuel_acetone":0.1,"lam":0.8,"stub_dir":0.0,"stub_yaw":0.0,"stub_pitch":0.0,"onb_mic":1.0,"onb_aim":0.0,"onb_el":30.0,"onb_hp":0.0,"gnd_sigma":0.0,"da_sigma":0.0,"far_sigma":30000000.0,"half_deg":0.0,"stub_track":0.0,"da_lp":0.0,"out_hp":30.0,"da_az":-3.0,"far_az":12.0,"tg_x":-2.0,"tg_y":0.0,"tg_z":0.95,"far_dl":0.0,"bank_dx":-0.0375,"onb_ground":0.0,"tun_w":4.5,"tun_h":5.5,"tun_cw":0.3,"tun_cc":0.0,"chg_lin":0.17,"onb_x":-1.075,"onb_y":0.2,"onb_z":0.88,"pad_sep":1.0,"gnd_fc":0.0,"sh_fc":1500,"sh_db":-3,"idle_map":0.21668471290867022,"ov_map":0.0509,"mech_mod":0.6,"ab_on":1.0,"ab_xlim":0.5,"ab_xw":0.25,"ab_eff":0.675,"ab_tau":0.0015,"ab_mouth":0.5,"fl_on":1.0,"fl_eff":0.2,"fl_turb":0.5,"fl_fc":250.0,"k_comb":0.0,"comb_x":1.0,"k_vib":0.3,"comb_lo":650.0,"rev_mix":1.0},"geom":{"bore":0.04953,"stroke":0.04826,"cr":7.8,"conrod":0.096,"dv":0.0215,"lmax":0.0062,"evo":102.01096601086984,"dur":290.0,"camx":1.4767886436763162,"cd":0.65,"stub_d":0.034,"drive_ratio":4.15,"blower_lobes":2,"blade_z":[12,12],"pr_max":5.696214039276613,"idle_rpm":2500.0,"redline":12000.0,"stub_y":0.45,"stub_z":0.25,"tail_x":-1.25,"tail_y":0.45,"tail_z":0.15,"b_mirror":1},"q":{"RG":312.6611364809158,"GAM":1.2939061828269274,"TFIRE":1108.1818469895627,"KEVO":5.30217447073139,"TSTUB0":470.60064178609315,"TSTUBA":331.1793757449951,"ABON":1.0,"XLIM":0.5,"XW":0.25,"ABEFF":0.675,"ABTAU":0.0015,"ABMOUTH":0.5,"AFRF":6.023670229007634,"LHVE":18384203.562340967,"TCH":445.6077429377809,"FLON":1.0,"FLK":0.29338643748870946,"FLTURB":0.5,"WRES":0.35,"BRES":0.0,"RRES":0.25,"WOV":2.5,"SURGQ":0.2,"REVM":1.0}}};
    var C_AIR = 343.0, P_ATM = 101325.0;   // exhaust R / gamma come from each set (q.RG / q.GAM: 290 / 1.32 up to v9.6, fuel thermochemistry from v9.8)
    var DL = 512, DLM = 511, RING = 2048, RINGM = 2047, SDL = 4096, SDLM = 4095;
    var RAD_K = 1.2 / (4 * Math.PI);
    var TWO_PI = 2 * Math.PI;
    // v1.10.4 (Wim, 12 Nov 2026): non-engine noise removed from every v9.x set - the onboard cockpit wind (Wind9, PH.wind /
    // persp.onboard.wind) and the v9.11 onboard structure-borne pickup (k_vib, band noise added after the cabin filter) are zero
    // here, in the kernel, so a re-export of SETS cannot bring them back. The fitted values stay in SETS (reversible: set to 1).
    // The wind noise draws still run (rng2 is shared with the valve-train noise), so the valve-train noise is unchanged.
    var NON_ENGINE_NOISE = 0;

    // ---------------------------------------------------------------- Butterworth (scipy.signal.butter, bilinear, prewarped) as biquads
    function cmul(a, b) { return [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]; }
    function cdiv(a, b) { var d = b[0] * b[0] + b[1] * b[1]; return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d]; }
    function csqrt(a) { var r = Math.sqrt(Math.sqrt(a[0] * a[0] + a[1] * a[1])), t = Math.atan2(a[1], a[0]) / 2; return [r * Math.cos(t), r * Math.sin(t)]; }
    /** returns an array of sections [b0,b1,b2,a1,a2] (a0 = 1), like sos rows */
    function butter(N, f, type, fs) {
      var k, i, p = [], z = [], gain = 1, fs2 = 2 * fs;
      for (k = 1; k <= N; k++) { var th = Math.PI * (2 * k + N - 1) / (2 * N); p.push([Math.cos(th), Math.sin(th)]); }
      if (type === 'bandpass') {
        var w1 = fs2 * Math.tan(Math.PI * f[0] / fs), w2 = fs2 * Math.tan(Math.PI * f[1] / fs), bw = w2 - w1, w0 = Math.sqrt(w1 * w2), pb = [];
        for (i = 0; i < N; i++) {
          var ph = [p[i][0] * bw / 2, p[i][1] * bw / 2], dsc = csqrt([ph[0] * ph[0] - ph[1] * ph[1] - w0 * w0, 2 * ph[0] * ph[1]]);
          pb.push([ph[0] + dsc[0], ph[1] + dsc[1]]); pb.push([ph[0] - dsc[0], ph[1] - dsc[1]]);
        }
        p = pb; for (i = 0; i < N; i++) z.push([0, 0]); gain = Math.pow(bw, N);
      } else {
        var wn = fs2 * Math.tan(Math.PI * f / fs);
        if (type === 'low') { for (i = 0; i < N; i++) p[i] = [p[i][0] * wn, p[i][1] * wn]; gain = Math.pow(wn, N); }
        else { for (i = 0; i < N; i++) { p[i] = cdiv([wn, 0], p[i]); z.push([0, 0]); } gain = 1; }
      }
      // bilinear
      var num = [1, 0], den = [1, 0], zd = [], pd = [];
      for (i = 0; i < z.length; i++) { num = cmul(num, [fs2 - z[i][0], -z[i][1]]); zd.push(cdiv([fs2 + z[i][0], z[i][1]], [fs2 - z[i][0], -z[i][1]])); }
      for (i = 0; i < p.length; i++) { den = cmul(den, [fs2 - p[i][0], -p[i][1]]); pd.push(cdiv([fs2 + p[i][0], p[i][1]], [fs2 - p[i][0], -p[i][1]])); }
      while (zd.length < pd.length) zd.push([-1, 0]);
      var kd = gain * cdiv(num, den)[0];
      // group poles into conjugate pairs / reals, zeros likewise (real zeros only here: +1 / -1 / 0)
      var used = [], pairs = [];
      for (i = 0; i < pd.length; i++) {
        if (used[i]) continue; used[i] = true;
        if (Math.abs(pd[i][1]) > 1e-12) {
          var best = -1, bd = 1e9;
          for (var j = i + 1; j < pd.length; j++) if (!used[j]) { var dd = Math.abs(pd[j][0] - pd[i][0]) + Math.abs(pd[j][1] + pd[i][1]); if (dd < bd) { bd = dd; best = j; } }
          used[best] = true; pairs.push([2 * pd[i][0], pd[i][0] * pd[i][0] + pd[i][1] * pd[i][1], 2]);
        } else pairs.push([pd[i][0], 0, 1]);
      }
      // merge single real poles pairwise
      var secP = [], singles = [];
      for (i = 0; i < pairs.length; i++) { if (pairs[i][2] === 2) secP.push([-pairs[i][0], pairs[i][1]]); else singles.push(pairs[i][0]); }
      while (singles.length >= 2) { var r1 = singles.shift(), r2 = singles.shift(); secP.push([-(r1 + r2), r1 * r2]); }
      if (singles.length) secP.push([-singles[0], 0, 1]);
      var zr = zd.map(function (q) { return q[0]; }), secs = [];
      for (i = 0; i < secP.length; i++) {
        var order = secP[i].length > 2 ? 1 : 2, b;
        if (order === 2) { var za = zr.shift(), zb = zr.shift(); b = [1, -(za + zb), za * zb]; }
        else { var zc = zr.shift(); b = [1, -zc, 0]; }
        if (i === 0) b = [b[0] * kd, b[1] * kd, b[2] * kd];
        secs.push([b[0], b[1], b[2], secP[i][0], secP[i][1]]);
      }
      return secs;
    }
    function Filt(secs) { this.c = secs; this.n = secs.length; this.s = new Float64Array(2 * secs.length); }
    Filt.prototype.f = function (x) {   // transposed direct form II (sosfilt)
      var c = this.c, s = this.s, i, k, y;
      for (i = 0; i < this.n; i++) {
        k = c[i]; y = k[0] * x + s[2 * i];
        s[2 * i] = k[1] * x - k[3] * y + s[2 * i + 1];
        s[2 * i + 1] = k[2] * x - k[4] * y;
        x = y;
      }
      return x;
    };

    // ---------------------------------------------------------------- xorshift64 (numba _rnd), as two uint32 halves
    function Rng(seedHi, seedLo) { this.h = seedHi >>> 0; this.l = seedLo >>> 0; }
    Rng.prototype.rnd = function () {
      var h = this.h, l = this.l, th, tl;
      th = ((h << 13) | (l >>> 19)) >>> 0; tl = (l << 13) >>> 0; h = (h ^ th) >>> 0; l = (l ^ tl) >>> 0;
      tl = ((l >>> 7) | (h << 25)) >>> 0; th = h >>> 7; h = (h ^ th) >>> 0; l = (l ^ tl) >>> 0;
      th = ((h << 17) | (l >>> 15)) >>> 0; tl = (l << 17) >>> 0; h = (h ^ th) >>> 0; l = (l ^ tl) >>> 0;
      this.h = h; this.l = l;
      return (h * 2097152 + (l >>> 11)) * (1.0 / 9007199254740992.0);
    };
    Rng.prototype.gau = function () { return (this.rnd() + this.rnd() + this.rnd() + this.rnd() - 2.0) * 1.732; };
    Rng.prototype.norm = function () {   // Box-Muller (post-chain noise sources; numpy RandomState is not reproduced there)
      var u = this.rnd(), v = this.rnd(); if (u < 1e-300) u = 1e-300;
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(TWO_PI * v);
    };
    // seed = 0x9E3779B97F4A7C15 ^ (seed * 7919) (seed * 7919 < 2^32 for small seeds)
    function rngFor(seed) { return new Rng(0x9E3779B9, (0x7F4A7C15 ^ (seed * 7919)) >>> 0); }

    // v9.8 mic gradient term -i / (k r) as run by brm9.py (centred, Hann-windowed 1024-tap FIR at 44.1 kHz, non-causal): causal
    // stand-in G(s) = (s + zr) / (s^2 + (w0 / Q) s + w0^2) x c / r (an integrator above ~150 Hz, rolled off like the window below);
    // tuned on test/node_compare_v9.js against the Python renders (onboard: 1/3-oct. 0.05 dB per frame). HANN0 = window centre value.
    var MICG = { f0: 15.0, q: 0.6, zr: 120.0 }, HANN0 = 0.5 - 0.5 * Math.cos(2 * Math.PI * 512 / 1023);
    function micBiquad(fs) {   // bilinear transform, sections [b0, b1, b2, a1, a2]
      var K = 2 * fs, w0 = 2 * Math.PI * MICG.f0, bq = w0 / MICG.q, zr = MICG.zr, a0 = K * K + bq * K + w0 * w0;
      return [(K + zr) / a0, 2 * zr / a0, (zr - K) / a0, (2 * w0 * w0 - 2 * K * K) / a0, (K * K - bq * K + w0 * w0) / a0];
    }
    var SOURCES = { exhL: [0.0, 0.45, 0.25], exhR: [0.0, -0.45, 0.25], blower: [1.1, 0.0, 0.5], mech: [0.35, 0.0, 0.55] };
    function dist3(a, b) { var dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2]; return Math.sqrt(dx * dx + dy * dy + dz * dz); }
    function interp(x, X, Y) {
      if (x <= X[0]) return Y[0];
      for (var i = 1; i < X.length; i++) if (x <= X[i]) return Y[i - 1] + (x - X[i - 1]) / (X[i] - X[i - 1]) * (Y[i] - Y[i - 1]);
      return Y[Y.length - 1];
    }
    function pmod(a, m) { var r = a % m; return r < 0 ? r + m : r; }

    // ---- shared kernel / synth classes (one hidden class for every model, so switching models keeps the JIT code monomorphic);
    //      the per-model data (parameter set, firing / stub geometry, perspectives) comes in as MD (build below)
    // ---------------------------------------------------------------- the model at internal rate fs: kernel + post chain
    function Eng(MD, fs, pp, seed) {
      this.MD = MD; var MD = this.MD, P = MD.P, PH = MD.PH, GE = MD.GE, QD = MD.QD, FIRE = MD.FIRE, BANK = MD.BANK, TWW = MD.TWW, CDN = MD.CDN, POS = MD.POS, stubXyz = MD.stubXyz, perspective = MD.perspective;
      this.fs = fs; this.pp = pp;
      var a = GE.stub_d / 2, i, c;
      this.AS = Math.PI * a * a; this.ENDC = a;
      this.thrK = 1.0 - Math.exp(-1.0 / (0.03 * fs)); this.boostK = 1.0 - Math.exp(-1.0 / (0.25 * fs)); this.tempK = 1.0 - Math.exp(-1.0 / (1.5 * fs));
      this.avMax = GE.cd * Math.PI * GE.dv * GE.lmax; this.aPort = GE.cd * 0.85 * Math.PI * GE.dv * GE.dv / 4;
      this.vd = 0.25 * Math.PI * GE.bore * GE.bore * GE.stroke; this.vc = this.vd / (GE.cr - 1.0);
      this.slen = new Float64Array(16); this.cyl6 = new Float64Array(16); this.fire = new Float64Array(16);
      for (c = 1; c <= 16; c++) {
        i = c - 1;
        this.slen[i] = Math.max(PH.stub_len * (1.0 + PH.stub_sig * P.stub_z[i] + PH.stub_c * CDN[i]) + 0.61 * a, 0.1);
        this.cyl6[i] = Math.max(1.0 + PH.chg_c * CDN[i], 0.05) * Math.max(1.0 + PH.chg_lin * ((3.5 - POS[i]) / 3.5), 0.05);
        this.fire[i] = FIRE[i] - 0.5 * (PH.half_deg || 0) * (POS[i] < 4 ? 1.0 : -1.0);
      }
      this.cyl6[PH.weak_cyl - 1] = PH.weak;
      // kernel state
      this.open = new Uint8Array(16); this.mass = new Float64Array(16); this.ise = new Float64Array(16); this.ee = new Float64Array(16);
      for (i = 0; i < 16; i++) this.ee[i] = 1.0;
      // v9.9 (cyl[:, 3, 4, 7..12] in brm9._k9): burnt share, unburnt fuel of this cycle, last mouth source, fuel stored in the stub,
      // afterburn countdown (samples) / energy, mouth-flame outflow lag / noise state
      this.abB = new Float64Array(16); this.abU = new Float64Array(16); this.qm7 = new Float64Array(16); this.abS = new Float64Array(16);
      this.abN = new Float64Array(16); this.abE = new Float64Array(16); this.flM = new Float64Array(16); this.flZ = new Float64Array(16);
      this.flA = 1.0 - Math.exp(-2.0 * Math.PI * PH.fl_fc / fs); this.flMk = 1.0 - Math.exp(-1.0 / (0.001 * fs));
      this.AT = new Float64Array(16 * DL); this.AF = new Float64Array(16 * DL); this.rptr = new Int32Array(16);
      this.B = new Float64Array(16 * DL); this.widx = new Int32Array(16);
      this.lp = new Float64Array(16); this.uprev = new Float64Array(16);
      this.ring0 = new Float64Array(RING); this.ring1 = new Float64Array(RING); this.ridx = 0;
      this.TH = 0; this.THR = 0; this.BOOST = 1.0; this.TSTUB = PH.t_stub0 + 0.3 * PH.t_stuba;   /* brm9: start value from the unscaled phys */ this.OMD = 0; this.EM = 1.0; this.BPH = 0; this.NSEED = 0;
      this.rng = rngFor(seed || 9); this.rng2 = rngFor(41 + (seed || 9));
      this._r = null; this._t = 0;
      this.setPersp(pp);
      // post chain (brm9._render)
      var f = this.fs, lim = 0.45 * f;
      function F(N, fc, t) { return new Filt(butter(N, fc, t, f)); }
      function cl(x) { return Math.min(x, lim); }
      this.jetL = F(2, [1500, cl(9000)], 'bandpass'); this.jetR = F(2, [1500, cl(9000)], 'bandpass');
      this.blF = F(1, [400, cl(PH.whine_lp)], 'bandpass'); this.rushF = F(2, [2000, cl(8000)], 'bandpass');
      this.mechF = F(2, [1000, cl(7000)], 'bandpass'); this.mechPh = 0;
      this.windF = F(2, [Math.min(PH.wind_fc, 9000), cl(10000)], 'bandpass'); this.gust = 0; this.gustN = 0; this.gustBlk = Math.max(1, Math.round(256 * f / 44100));
      this.speed = 0;
      // v9.11 onboard structure-borne pickup (brm9._render comb x k_vib): charge-pressure share low-passed at 200 Hz, band noise
      // comb_lo-6 kHz, firing-rate (8 per rev) modulation. Own noise generator, so the older sets' mech / wind noise is untouched.
      this.vibLc = F(1, cl(200), 'low'); this.vibF = F(2, [Math.min(PH.comb_lo !== undefined ? PH.comb_lo : 1000.0, 0.4 * f), cl(6000)], 'bandpass');
      this.vibPh = 0; this.rng3 = rngFor(77 + (seed || 9));
    }
    Eng.prototype.setPersp = function (pp) {
      var MD = this.MD, P = MD.P, PH = MD.PH, GE = MD.GE, QD = MD.QD, FIRE = MD.FIRE, BANK = MD.BANK, TWW = MD.TWW, CDN = MD.CDN, POS = MD.POS, stubXyz = MD.stubXyz, perspective = MD.perspective;
      var fs = this.fs, c, i, e, lim = 0.45 * fs;
      this.pp = pp;
      // per-stub relative delay / gain seen from the mean ear (stub_geometry)
      var m = [0, 0, 0];
      for (e = 0; e < pp.ears.length; e++) for (i = 0; i < 3; i++) m[i] += pp.ears[e][i] / pp.ears.length;
      this.rdly = new Float64Array(16); this.rgain = new Float64Array(16);
      var dmin = 1e9;
      for (c = 1; c <= 16; c++) {
        var src = SOURCES[c <= 8 ? 'exhL' : 'exhR'], r0 = dist3(src, m), ri = dist3(stubXyz(c), m);
        this.rgain[c - 1] = r0 / ri; this.rdly[c - 1] = (ri - r0) / C_AIR * fs; if (this.rdly[c - 1] < dmin) dmin = this.rdly[c - 1];
      }
      for (i = 0; i < 16; i++) this.rdly[i] = this.rdly[i] - dmin + 1.0;
      // Spatializer taps: per source, ear, image -> integer delay and the direct / air-absorbed weights
      var names = ['exhL', 'exhR', 'blower', 'mech'];
      this.taps = [];
      var shadowOn = pp.shadow_far_db < 0;
      // v9.4 tunnel images (onboard reference filmed in a road tunnel; used by the v9.9 fit) and the v9.8 onboard M-S mic
      var tunOn = !pp.outside && PH.tun_cw > 0, micOn = !pp.outside && PH.onb_mic > 0, mic = null;
      if (micOn) { var az = PH.onb_aim * Math.PI / 180, el = PH.onb_el * Math.PI / 180; mic = [Math.cos(az) * Math.cos(el), Math.sin(az) * Math.cos(el), Math.sin(el)]; }
      this.micOn = micOn;
      this.far = [false, false, false, false];
      for (var s = 0; s < 4; s++) {
        var sp = SOURCES[names[s]], list = [];
        this.far[s] = shadowOn && ((names[s] === 'exhR' && pp.ears[0][1] > 0) || (names[s] === 'exhL' && pp.ears[0][1] < 0));
        for (e = 0; e < 2; e++) {
          var ear = pp.ears[e], imgs = [[sp[1], sp[2], 1.0], [sp[1], -sp[2], pp.ground]];
          if (tunOn) imgs.push([2 * PH.tun_w - sp[1], sp[2], PH.tun_cw], [-2 * PH.tun_w - sp[1], sp[2], PH.tun_cw], [sp[1], 2 * PH.tun_h - sp[2], PH.tun_cc]);
          for (var q = 0; q < imgs.length; q++) {
            var gz = imgs[q][2]; if (gz === 0) continue;
            var r = dist3([sp[0], imgs[q][0], imgs[q][1]], ear) + ((this.far[s] && !pp.outside) ? PH.far_dl : 0);
            var wf = Math.min(1, Math.max(0, r / 80.0)), gg = gz / Math.max(r, 0.3);
            if (micOn && q === 0) {
              // v9.8 VP88 mid capsule (direct paths only): 0.5 + 0.5 cos(theta) (1 - i / kr). Python applies it as a centred
              // 1024-tap FIR (1 sample net delay); here: pressure part exact, gradient part (c / r) x G(s) on the source streams
              var rv = [sp[0] - ear[0], imgs[q][0] - ear[1], imgs[q][1] - ear[2]], rr = Math.sqrt(rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2]);
              var ct = (rv[0] * mic[0] + rv[1] * mic[1] + rv[2] * mic[2]) / Math.max(rr, 1e-6), pw = HANN0 * (0.5 + 0.5 * ct), gw = 0.5 * ct * C_AIR / Math.max(rr, 0.05);
              list.push([e, Math.round(r / C_AIR * fs) + 1, (1 - wf) * gg * pw, wf * gg * pw, (1 - wf) * gg * gw, wf * gg * gw]);
            } else list.push([e, Math.round(r / C_AIR * fs), (1 - wf) * gg, wf * gg]);
          }
        }
        this.taps.push(list);
      }
      this.useShadow = PH.sh_fc > 0;
      this.gsh = this.useShadow ? Math.pow(10, PH.sh_db / 20) : Math.pow(10, pp.shadow_far_db / 20);
      this.shLo = [new Filt(butter(1, this.useShadow ? PH.sh_fc : 1400, 'low', fs)), new Filt(butter(1, this.useShadow ? PH.sh_fc : 1400, 'low', fs))];
      if (!this.useShadow) this.shLo = [new Filt(butter(2, 1400, 'low', fs)), new Filt(butter(2, 1400, 'low', fs))];
      this.air = []; this.dlS = []; this.dlA = [];
      for (s = 0; s < 4; s++) { this.air.push(new Filt(butter(1, 2500, 'low', fs))); this.dlS.push(new Float64Array(SDL)); this.dlA.push(new Float64Array(SDL)); }
      this.micG = []; this.micGA = []; this.dlG = []; this.dlGA = [];
      if (micOn) for (s = 0; s < 4; s++) { this.micG.push(new Filt([micBiquad(fs)])); this.micGA.push(new Filt([micBiquad(fs)])); this.dlG.push(new Float64Array(SDL)); this.dlGA.push(new Float64Array(SDL)); }
      this.dli = 0;
      // recording chain
      if (pp.outside) {
        this.chA = [new Filt(butter(2, Math.min(PH.out_lp, lim), 'low', fs)), new Filt(butter(2, Math.min(PH.out_lp, lim), 'low', fs))]; this.chMix = PH.out_mix;
        this.chH = PH.out_hp > 0 ? [new Filt(butter(2, PH.out_hp, 'high', fs)), new Filt(butter(2, PH.out_hp, 'high', fs))] : null;
      } else {
        this.chA = PH.onb_mix > 0 ? [new Filt(butter(2, Math.min(PH.onb_lp, lim), 'low', fs)), new Filt(butter(2, Math.min(PH.onb_lp, lim), 'low', fs))] : null; this.chMix = PH.onb_mix;
        this.chH = PH.onb_hp > 0 ? [new Filt(butter(2, PH.onb_hp, 'high', fs)), new Filt(butter(2, PH.onb_hp, 'high', fs))] : null;   // v9.8 VP88 low-cut (0 in the fits)
      }
      this.windOn = pp.wind > 0;
    };
    Eng.prototype.kimp = function (rpm, thr) {
      var MD = this.MD, P = MD.P, PH = MD.PH, GE = MD.GE, QD = MD.QD, FIRE = MD.FIRE, BANK = MD.BANK, TWW = MD.TWW, CDN = MD.CDN, POS = MD.POS, stubXyz = MD.stubXyz, perspective = MD.perspective;
      var w = Math.max(rpm, 500.0) * TWO_PI / 60;
      var T = interp(rpm, P.mays_rpm, P.mays_bhp) * 745.7 / w * Math.pow(Math.max(thr, 0.05), 1.3);
      return T * (4 * Math.PI / w) / 16.0 / (P.j_crank * w);
    };
    /** n internal samples; rpm / thr ramp linearly from the previous call's values (BRM9Synth.render). L/R += output */
    Eng.prototype.run = function (n, rpmT, thrT, L, R, gain) {
      var MD = this.MD, P = MD.P, PH = MD.PH, GE = MD.GE, QD = MD.QD, FIRE = MD.FIRE, BANK = MD.BANK, TWW = MD.TWW, CDN = MD.CDN, POS = MD.POS, stubXyz = MD.stubXyz, perspective = MD.perspective;
      var r0 = this._r !== null ? this._r : rpmT, t0 = this._r !== null ? this._t : thrT;
      this._r = rpmT; this._t = thrT;
      var tq = this.kimp(rpmT, thrT);
      var fs = this.fs, dt = 1.0 / fs, g = QD.GAM, R_GAS = QD.RG, gm1 = g - 1.0, crit = Math.pow((g + 1.0) / 2.0, g / gm1);
      var psiCh = Math.sqrt(g) * Math.pow(2.0 / (g + 1.0), (g + 1.0) / (2.0 * gm1));
      var AS = this.AS, vc = this.vc, bore = GE.bore, stroke = GE.stroke, conrod = GE.conrod, rcr = 0.5 * stroke, apist = 0.25 * Math.PI * bore * bore;
      var IDLE = GE.idle_rpm, RED = GE.redline, PRMAX = GE.pr_max, IDLEMAP = PH.idle_map, OVMAP = PH.ov_map, TS0 = QD.TSTUB0, TSA = QD.TSTUBA;
      var EVO = GE.evo, DUR = GE.dur, CAMX = GE.camx, KEVO = QD.KEVO, KMOT = PH.k_mot, TFIRE = QD.TFIRE, TMOT = PH.t_mot;
      var KNL = PH.k_nl, ROLOSS = PH.ro_loss, COV = PH.cov, COVI = PH.cov_idle, MISF = PH.misfire, MISI = PH.misfire_idle, TORS = PH.tors_deg;
      var DR = GE.drive_ratio, LOBES = GE.blower_lobes, WW = PH.whine_w, BZ1 = GE.blade_z[0], BZ2 = GE.blade_z[1], KWH = PH.whine, KBL = PH.blade, KRU = PH.rush;
      // v9.10 blower split (brm9.py Q_WRES ... Q_SURGQ; sets without them = v9.9 kernel defaults, bit-identical audio)
      var WRES = QD.WRES !== undefined ? QD.WRES : 0.0, BRES = QD.BRES !== undefined ? QD.BRES : 0.0, RRES = QD.RRES !== undefined ? QD.RRES : 0.25;
      var WOV = QD.WOV !== undefined ? QD.WOV : 0.0, SURGQ = QD.SURGQ !== undefined ? Math.max(QD.SURGQ, 1e-3) : 0.25;
      // v9.11 reversion energy balance (brm9.py Q_REVM; sets without it = kernel default 0 = v9.10, bit-identical audio)
      var REVM = QD.REVM !== undefined ? QD.REVM : 0.0;
      var avMax = this.avMax, aPort = this.aPort, evc = EVO + DUR, beta = KNL * 0.5 * (g + 1.0), invRs = 1.0 / R_GAS;
      var open = this.open, mass = this.mass, ise = this.ise, ee = this.ee, cyl6 = this.cyl6, fire = this.fire, slen = this.slen, rdly = this.rdly, rgain = this.rgain;
      var AT = this.AT, AF = this.AF, rptr = this.rptr, B = this.B, widx = this.widx, lpS = this.lp, uprev = this.uprev, ring0 = this.ring0, ring1 = this.ring1;
      var rng = this.rng, rng2 = this.rng2, jetC = 2.0 * Math.sqrt(AS / Math.PI) / (343.0 * 343.0), kjet4 = PH.k_jet * 4.0;
      var pp = this.pp, taps = this.taps, far = this.far, air = this.air, dlS = this.dlS, dlA = this.dlA, gsh = this.gsh, shLo = this.shLo, useSh = this.useShadow;
      var chA = this.chA, chH = this.chH, chMix = this.chMix, mechK = PH.mech * 300.0, mechMod = PH.mech_mod, mechOn = PH.mech > 0;
      var rrB = rpmT / RED, mechLev = mechOn ? mechK * Math.pow(rrB, 1.5) : 0, mechInc = rpmT / 60.0 / fs;
      var windLev = NON_ENGINE_NOISE ? PH.wind * 300.0 : 0.0, u3 = Math.pow(Math.max(0, this.speed * 3.6) / 200.0, 3);
      // v9.11 onboard structure-borne pickup (k_vib; 0 / absent = off up to v9.10): 300 x lc^comb_x x rpm/redline x env x band noise,
      // added to the onboard mic signal after the mic chain and the wind (brm9._render), both channels, never outside
      var KVIB = NON_ENGINE_NOISE && PH.k_vib > 0 ? PH.k_vib : 0.0, vibK = KVIB > 0.0, vibOn = vibK && !pp.outside, vibLev = vibOn ? KVIB * 300.0 * rrB : 0.0;
      var COMBX = PH.comb_x !== undefined ? PH.comb_x : 1.0, rng3 = this.rng3, chs = 0.0;
      var src = [0, 0, 0, 0], yo = [0, 0], micOn = this.micOn, micG = this.micG, micGA = this.micGA, dlG = this.dlG, dlGA = this.dlGA;
      // v9.9 overrun afterburn / rich-exhaust mouth flame (off = 0 up to v9.8)
      var ABON = QD.ABON > 0, FLON = QD.FLON > 0, MOUTH = ABON || FLON, CR = GE.cr, XLIM = QD.XLIM, XW = QD.XW, ABEFF = QD.ABEFF, LHVE = QD.LHVE;
      var ABTAU = QD.ABTAU, ABTS = QD.ABTAU * fs, ABG = gm1 / (g * P_ATM), ABMOUTH = QD.ABMOUTH, AFRF = QD.AFRF, TCH = QD.TCH;
      var FLK = QD.FLK, FLTURB = QD.FLTURB, FLA = this.flA, FLM = this.flMk, FLN = FLON ? Math.sqrt((2.0 - FLA) / FLA) : 0, vdc = this.vd + vc;
      var abB = this.abB, abU = this.abU, qm7 = this.qm7, abS = this.abS, abN = this.abN, abE = this.abE, flM = this.flM, flZ = this.flZ;
      for (var k = 0; k < n; k++) {
        var tt0 = (k + 1) / n, rpm_t = r0 + (rpmT - r0) * tt0, thr_in = t0 + (thrT - t0) * tt0;
        var thr = this.THR + (thr_in - this.THR) * this.thrK; this.THR = thr;
        var rr = rpm_t / RED, pr = 1.0 + (PRMAX - 1.0) * rr * rr;
        this.BOOST += (pr - this.BOOST) * this.boostK;
        var BOOST = this.BOOST, ov = 0.0;
        if (thr < 0.06) { ov = (rpm_t - IDLE * 1.3) / (IDLE * 0.9); ov = Math.min(1.0, Math.max(0.0, ov)) * (0.06 - thr) / 0.06; }
        var load = (IDLEMAP + (1.0 - IDLEMAP) * thr) * (1.0 - ov) + OVMAP * ov;
        var p_man = P_ATM * load * BOOST;
        chs = load * BOOST / PRMAX;   // v9.11: charge-pressure share (brm9 K[:, 13], drives the structure-borne source)
        var ttg = TS0 + TSA * load * (0.4 + 0.6 * rr);
        this.TSTUB += (ttg - this.TSTUB) * this.tempK;
        var Ts = this.TSTUB, cs = Math.sqrt(g * R_GAS * Ts), rho_s = P_ATM / (R_GAS * Ts), Z0 = rho_s * cs / AS, sqRTs = Math.sqrt(R_GAS * Ts);
        var rpm = rpm_t * (1.0 + this.OMD), dth = rpm * 6.0 / fs;
        this.OMD *= 1.0 - dth / 720.0;
        var th0 = this.TH, th1 = th0 + dth;
        this.TH = pmod(th1, 720.0);
        var tors = TORS * load * BOOST / PRMAX, jet0 = 0.0, jet1 = 0.0, NS = this.NSEED;
        var ao = 1.0 - Math.exp(-TWO_PI * (cs / (TWO_PI * this.ENDC)) / fs);
        var ll = 1.0 - load; if (ll < 0.0) ll = 0.0;
        for (var i = 0; i < 16; i++) {
          var a0 = pmod(th0 - fire[i] + tors * TWW[i], 720.0), a1 = a0 + dth, e;
          if (a1 >= 720.0) {
            a1 -= 720.0;
            e = 1.0 + (COV + COVI * ll * ll) * rng.gau();
            e *= cyl6[i];
            if (rng.rnd() < MISF * (0.3 + 0.7 * rr) * (0.2 + 0.8 * thr) + MISI * ll * ll) e = 0.1 * rng.rnd();
            if (ABON) {
              // v9.9: residual-gas dilution at this manifold pressure -> flame-kernel survival; unburnt charge
              var pmx = p_man > 2000.0 ? p_man : 2000.0, xr = Math.pow(P_ATM / pmx, 1.0 / g) / CR;
              if (xr > 1.0) xr = 1.0;
              var bm = 1.0 - (xr - XLIM) / XW, bb = 1.0;
              if (bm < 1.0) { if (rng.rnd() >= bm) bb = 0.5 * rng.rnd(); }
              e *= bb; abB[i] = bb;
              abU[i] = (1.0 - bb) * (pmx * vdc * (1.0 - xr) / (287.0 * TCH)) / AFRF;
            }
            if (e < 0.0) e = 0.0;
            ee[i] = e;
            this.EM += (e - this.EM) * 0.02;
            this.OMD += tq * (e - this.EM);
          }
          var q = 0.0, aeff = 0.0, V, ang, sn, x;
          if (a0 < EVO && a1 >= EVO) {
            e = ee[i];
            var pe = p_man * (KMOT + (KEVO - KMOT) * e), Te = TMOT + (TFIRE - TMOT) * Math.min(e, 1.2);
            ang = a1 * Math.PI / 180.0; sn = Math.sin(ang);
            V = vc + apist * (rcr * (1.0 - Math.cos(ang)) + conrod - Math.sqrt(conrod * conrod - rcr * rcr * sn * sn));
            var m0 = pe * V / (R_GAS * Te), rho0 = m0 / V;
            open[i] = 1; mass[i] = m0; ise[i] = pe / Math.pow(rho0, g);
            if (ABON) {
              // v9.9: this cycle's gas pushes the older stub contents out (volume ratio); its hot gas ignites what is left if it
              // burnt; its own unburnt charge (if any) is stored at the valve end
              var vex = m0 * R_GAS * Ts / P_ATM, vst = AS * slen[i], fx = vex / vst;
              if (fx > 1.0) fx = 1.0;
              var res = abS[i] * (1.0 - fx);
              if (abB[i] > 0.5 && res > 0.0) { abE[i] = ABEFF * res * LHVE; abN[i] = ABTS; res = 0.0; }
              abS[i] = res + abU[i] * (vex > vst ? vst / vex : 1.0); abU[i] = 0.0;
            }
          }
          if (open[i]) {
            if (a1 >= evc || a1 < EVO) open[i] = 0;
            else {
              x = (a1 - EVO) / DUR;
              var lift = Math.pow(Math.sin(Math.PI * x), CAMX);
              aeff = Math.min(avMax * lift, aPort);
              ang = (a1 % 360.0) * Math.PI / 180.0; sn = Math.sin(ang);
              V = vc + apist * (rcr * (1.0 - Math.cos(ang)) + conrod - Math.sqrt(conrod * conrod - rcr * rcr * sn * sn));
              var m = mass[i], qs = 0.0;
              for (var sub = 0; sub < 2; sub++) {
                var rho = m / V, p = ise[i] * Math.pow(rho, g), T = p / (rho * R_GAS), psi, md, rho_e, y;
                if (p >= P_ATM) {
                  var rp = p / P_ATM;
                  if (rp > crit) psi = psiCh;
                  else { y = 1.0 / rp; psi = Math.sqrt(Math.max(0.0, 2.0 * g / gm1 * (Math.pow(y, 2.0 / g) - Math.pow(y, (g + 1.0) / g)))); }
                  md = aeff * p / Math.sqrt(R_GAS * T) * psi;
                  var Tex = T * Math.pow(P_ATM / p, gm1 / g);
                  rho_e = P_ATM / (R_GAS * Tex);
                } else {
                  y = p / P_ATM;
                  psi = Math.sqrt(Math.max(0.0, 2.0 * g / gm1 * (Math.pow(y, 2.0 / g) - Math.pow(y, (g + 1.0) / g))));
                  md = -aeff * P_ATM / sqRTs * psi;
                  rho_e = rho_s;
                }
                var dm = md * 0.5 * dt;
                if (dm > 0.4 * m) { dm = 0.4 * m; md = dm / (0.5 * dt); }
                if (md < 0.0 && m - dm > 4.0 * P_ATM * V / (R_GAS * Ts)) { dm = 0.0; md = 0.0; }
                m -= dm;
                if (md < 0.0) {
                  // back-flow of stub gas: re-set the isentrope to the mixed state; v9.11 (REVM, 0 = v9.10): energy balance of the
                  // inflow, the stub gas brings its enthalpy -> dp = gamma R Ts dm_in / V (the cylinder is no longer a bottomless sink)
                  var rho2 = m / V, p2 = REVM > 0.0 ? p + REVM * g * R_GAS * Ts * (-dm) / V : p;
                  ise[i] = Math.max(p2, 0.2 * P_ATM) / Math.pow(rho2, g);
                }
                qs += 0.5 * md / rho_e;
              }
              mass[i] = m;
              q = qs;
            }
          }
          var qab = 0.0;
          if (abN[i] > 0.0) {
            // v9.9 afterburn heat release (raised cosine over ab_tau) -> volume source (gamma-1)/(gamma p) dQ/dt
            qab = ABG * abE[i] / ABTAU * (1.0 - Math.cos(TWO_PI * (1.0 - abN[i] / ABTS)));
            abN[i] -= 1.0;
            q += (1.0 - ABMOUTH) * qab;
          }
          // ---- stub waveguide i
          var o = i * DL, Dn = slen[i] / cs * fs, idx = widx[i];
          var di = Dn | 0, fr = Dn - di, ba = B[o + ((idx - di) & DLM)], bb = B[o + ((idx - di - 1) & DLM)], b_arr = ba + fr * (bb - ba);
          var rv = (AS - aeff) / (AS + aeff) * ROLOSS;
          var fin = Z0 * q + rv * b_arr, u = fin / (rho_s * cs);
          if (beta * u < -0.6 * cs) u = -0.6 * cs / beta;
          var ta = NS + slen[i] * fs / (cs + beta * u);
          if (ta > NS + DLM - 2) ta = NS + DLM - 2;
          var w = idx, tprev = AT[o + ((w - 1) & DLM)];
          if (ta < tprev + 0.02) ta = tprev + 0.02;
          AT[o + w] = ta; AF[o + w] = fin;
          var rpp = rptr[i];
          while (((rpp + 1) & DLM) !== w && AT[o + ((rpp + 1) & DLM)] <= NS) rpp = (rpp + 1) & DLM;
          rptr[i] = rpp;
          var t0a = AT[o + rpp], rn = (rpp + 1) & DLM, t1a = AT[o + rn], f_arr;
          if (rn === ((w + 1) & DLM) || t1a <= t0a) f_arr = AF[o + rpp];
          else {
            var fr0 = (NS - t0a) / (t1a - t0a);
            if (fr0 < 0.0) fr0 = 0.0; else if (fr0 > 1.0) fr0 = 1.0;
            f_arr = AF[o + rpp] + fr0 * (AF[o + rn] - AF[o + rpp]);
          }
          lpS[i] += ao * (f_arr - lpS[i]);
          var Mf = uprev[i] / AS / cs;
          if (Mf < 0.0) Mf = 0.0; else if (Mf > 0.9) Mf = 0.9;
          var rr2 = -ROLOSS * lpS[i] * (1.0 - Mf) / (1.0 + Mf), U = (f_arr - rr2) / Z0;
          B[o + idx] = rr2;
          widx[i] = (idx + 1) & DLM;
          var dU = (U - uprev[i]) * fs;
          uprev[i] = U;
          if (MOUTH) {
            var qm = ABMOUTH * qab;
            if (FLON) {
              // v9.9 mouth flame: heat release ~ outflow mass rate x h_c x fl_eff, ~1 ms mixing lag, band-limited turbulent fluctuation
              flM[i] += ((U > 0.0 ? rho_s * U : 0.0) - flM[i]) * FLM;
              flZ[i] += ((2.0 * rng.rnd() - 1.0) * 1.7320508 - flZ[i]) * FLA;
              qm += FLK * flM[i] * (1.0 + FLTURB * (flZ[i] * FLN));
            }
            dU += (qm - qm7[i]) * fs;
            qm7[i] = qm;
          }
          if (U > 0.0) {
            var uu = U / AS, u2 = uu * uu, jv = rho_s * u2 * u2 * jetC;
            if (BANK[i] === 0) jet0 += jv; else jet1 += jv;
          }
          var pos = this.ridx + rdly[i], pi_ = Math.floor(pos), fq = pos - pi_, vv = rgain[i] * dU, ring = BANK[i] === 0 ? ring0 : ring1;
          ring[pi_ & RINGM] += (1.0 - fq) * vv;
          ring[(pi_ + 1) & RINGM] += fq * vv;
        }
        var kr = this.ridx & RINGM, busL = ring0[kr], busR = ring1[kr];
        ring0[kr] = 0.0; ring1[kr] = 0.0; this.ridx++;
        var jn0 = jet0 * (2.0 * rng.rnd() - 1.0), jn1 = jet1 * (2.0 * rng.rnd() - 1.0);
        // ---- supercharger
        this.BPH += DR * rpm / 60.0 / fs;
        if (this.BPH > 1.0) this.BPH -= 1.0;
        var xl = (this.BPH * LOBES) % 1.0, pul = 0.0;
        if (xl < WW) pul = 0.5 - 0.5 * Math.cos(TWO_PI * xl / WW);
        pul -= 0.5 * WW;
        var dp = rr * rr * BOOST / PRMAX, flow = 0.25 + 0.75 * load;
        // v9.10: (1) SHAFT-SPEED tones (impellers geared 4.15:1 to the crank): share *_RES independent of the through-flow, x the
        // throttled-inlet low-flow rise (SU throttles ahead of the blower: strongest on the overrun, 0 above SURGQ load);
        // (2) FLOW noise (air rush) follows the through-flow and collapses on lift-off.  Defaults (0, 0, 0.25, 0) = v9.9 exactly.
        var wsh = WRES + (1.0 - WRES) * flow, bsh = BRES + (1.0 - BRES) * flow, rfl = RRES + (1.0 - RRES) * load;
        var srg = 1.0 + WOV * Math.max(0.0, 1.0 - load / SURGQ);
        wsh *= srg; bsh *= srg;
        var bl = KWH * pul * dp * wsh / WW, phs = TWO_PI * this.BPH;
        bl += KBL * dp * bsh * (Math.sin(BZ1 * phs) + 0.7 * Math.sin(BZ2 * phs + 1.0));
        var rush = KRU * dp * rr * rfl * (2.0 * rng.rnd() - 1.0);
        this.NSEED += 1.0;
        if (this.NSEED > 1.0e6) { this.NSEED -= 1.0e6; for (var j = 0; j < AT.length; j++) AT[j] -= 1.0e6; }
        // ---- post chain (brm9._render): buses
        src[0] = RAD_K * busL + kjet4 * this.jetL.f(jn0);
        src[1] = RAD_K * busR + kjet4 * this.jetR.f(jn1);
        src[2] = 30.0 * (this.blF.f(bl) + this.rushF.f(rush));
        if (mechOn) {
          this.mechPh += mechInc; if (this.mechPh > 1.0) this.mechPh -= 1.0;
          src[3] = mechLev * (1.0 + mechMod * Math.cos(16.0 * TWO_PI * this.mechPh)) * this.mechF.f(rng2.norm());
        } else src[3] = 0;
        // ---- Spatializer (static listener)
        var di2 = this.dli; yo[0] = 0; yo[1] = 0;
        for (var s = 0; s < 4; s++) {
          if (s === 3 && !mechOn) continue;
          var sg = src[s];
          if (far[s]) {
            if (useSh) { var lo = shLo[s].f(sg); sg = lo + gsh * (sg - lo); }
            else sg = gsh * (0.35 * sg + 0.65 * shLo[s].f(sg));
          }
          var sa = air[s].f(sg);
          dlS[s][di2] = sg; dlA[s][di2] = sa;
          var tl = taps[s];
          if (micOn) {
            var dG = dlG[s], dGA = dlGA[s];
            dG[di2] = micG[s].f(sg); dGA[di2] = micGA[s].f(sa);
            for (var t = 0; t < tl.length; t++) { var tp = tl[t], ix = (di2 - tp[1]) & SDLM; yo[tp[0]] += tp[2] * dlS[s][ix] + tp[3] * dlA[s][ix]; if (tp.length > 4) yo[tp[0]] += tp[4] * dG[ix] + tp[5] * dGA[ix]; }
          } else for (var t2 = 0; t2 < tl.length; t2++) { var tpn = tl[t2], ix2 = (di2 - tpn[1]) & SDLM; yo[tpn[0]] += tpn[2] * dlS[s][ix2] + tpn[3] * dlA[s][ix2]; }
        }
        this.dli = (di2 + 1) & SDLM;
        var yl = yo[0], yr = yo[1];
        if (chA) { yl = (1 - chMix) * yl + chMix * chA[0].f(yl); yr = (1 - chMix) * yr + chMix * chA[1].f(yr); }
        if (chH) { yl = chH[0].f(yl); yr = chH[1].f(yr); }
        if (this.windOn) {
          if (++this.gustN >= this.gustBlk) { this.gustN = 0; this.gust += (rng2.norm() * 0.3 - this.gust) * 0.05; }
          var wv = u3 * (1.0 + this.gust) * this.windF.f(rng2.norm()) * windLev;
          yl += wv; yr += 0.9 * wv;
        }
        if (vibK) {
          var lcv = this.vibLc.f(chs);
          if (vibOn) {
            this.vibPh += mechInc; if (this.vibPh > 1.0) this.vibPh -= 1.0;
            var vv = vibLev * Math.pow(lcv > 0.0 ? lcv : 0.0, COMBX) * (1.0 + mechMod * Math.cos(8.0 * TWO_PI * this.vibPh)) * this.vibF.f(rng3.norm());
            yl += vv; yr += vv;
          }
        }
        L[k] += gain * yl; R[k] += gain * yr;
      }
    };
    Eng.prototype.copyKernel = function (o) {   // same engine state (other listener)
      var MD = this.MD, P = MD.P, PH = MD.PH, GE = MD.GE, QD = MD.QD, FIRE = MD.FIRE, BANK = MD.BANK, TWW = MD.TWW, CDN = MD.CDN, POS = MD.POS, stubXyz = MD.stubXyz, perspective = MD.perspective;
      var keys = ['open', 'mass', 'ise', 'ee', 'AT', 'AF', 'rptr', 'B', 'widx', 'lp', 'uprev', 'ring0', 'ring1', 'abB', 'abU', 'qm7', 'abS', 'abN', 'abE', 'flM', 'flZ'];
      for (var i = 0; i < keys.length; i++) this[keys[i]].set(o[keys[i]]);
      var sc = ['ridx', 'TH', 'THR', 'BOOST', 'TSTUB', 'OMD', 'EM', 'BPH', 'NSEED', '_r', '_t', 'mechPh', 'gust', 'speed', 'vibPh'];
      for (i = 0; i < sc.length; i++) this[sc[i]] = o[sc[i]];
      this.rng.h = o.rng.h; this.rng.l = o.rng.l; this.rng2.h = o.rng2.h; this.rng2.l = o.rng2.l; this.rng3.h = o.rng3.h; this.rng3.l = o.rng3.l;
    };

    // ---------------------------------------------------------------- 2x interpolator (windowed-sinc half-band, odd phase)
    var HB_N = 12, HB = new Float64Array(2 * HB_N);
    (function () {
      var s = 0, k;
      for (k = 0; k < 2 * HB_N; k++) { var t = k - HB_N + 0.5, x = Math.PI * t; HB[k] = Math.sin(x) / x * (0.54 + 0.46 * Math.cos(Math.PI * t / (HB_N + 0.5))); s += HB[k]; }
      for (k = 0; k < 2 * HB_N; k++) HB[k] /= s;
    })();
    function Up2() { this.x = new Float64Array(4 * HB_N); this.i = 0; }
    Up2.prototype.push = function (v, out, o) {   // one input sample -> out[o], out[o+1]
      var x = this.x, N = 2 * HB_N, i = this.i, k, acc = 0;
      x[i] = v; x[i + N] = v; this.i = (i + 1) % N;
      var base = this.i;   // oldest
      for (k = 0; k < N; k++) acc += HB[k] * x[base + k];
      out[o] = x[base + HB_N - 1]; out[o + 1] = acc;
    };

    // ---------------------------------------------------------------- the synth (ListenerSource interface)
    function isMainThread() { return typeof registerProcessor !== 'function'; }
    function BrmSynthV9(MD, spec, listener, fs, opts) {
      this.MD = MD; var MD = this.MD, P = MD.P, PH = MD.PH, GE = MD.GE, QD = MD.QD, FIRE = MD.FIRE, BANK = MD.BANK, TWW = MD.TWW, CDN = MD.CDN, POS = MD.POS, stubXyz = MD.stubXyz, perspective = MD.perspective;
      opts = opts || {};
      this.fs = fs || 44100;
      this.spec = Object.assign({ idle_rpm: 3800, redline: 12000, name: 'BRM P30 V16' }, spec || {});
      this.listener = listener || {};
      this.pos = this.listener.position || this.listener.pos || 'rear';
      var qual = opts.quality || this.spec.quality || 'auto';
      this.auto = qual === 'auto';
      this.dec = qual === 'lite' ? 2 : (qual === 'hq' ? 1 : (isMainThread() ? 2 : 1));
      if (this.fs < 30000) this.dec = 1;
      this.seed = opts.seed || 9;
      this.eng = new Eng(MD, this.fs / this.dec, perspective(this.pos), this.seed);
      this.up = [new Up2(), new Up2()];
      this.bL = new Float64Array(64); this.bR = new Float64Array(64); this.uL = new Float64Array(2); this.uR = new Float64Array(2);
      this.fifoL = new Float64Array(256); this.fifoR = new Float64Array(256); this.fn = 0;
      this.gain = this.listener.level !== undefined ? this.listener.level : 1;   // (tests) raw model scale; the game level is gain_db in js/models.js
      this.cpuMs = 0; this.cpuAudio = 0; this.speed = 0; this.xf = null;
    }
    BrmSynthV9.prototype.ctl = function (m) { if (m && m.speed !== undefined) { this.speed = m.speed; this.eng.speed = m.speed; if (this.xf) this.xf.eng.speed = m.speed; } };
    BrmSynthV9.prototype.quality = function () { return this.dec === 1 ? 'hq' : 'lite'; };
    /** v1.10 engine bay: a thin read of the kernel state (no computation): crank angle TH (deg, 0..720), rpm with the torque-impulse
     *  ripple, smoothed throttle, boost, blower phase (revs), quality (1 = full rate), per-cylinder combustion energy ee, afterburn
     *  energy abE (changes on each v9.9 afterburn event) and mouth-flame outflow flM. o: Float32Array(56). */
    BrmSynthV9.prototype.bay = function (o) {
      var e = this.eng, i;
      o[0] = e.TH; o[1] = (e._r !== null ? e._r : 0) * (1.0 + e.OMD); o[2] = e.THR; o[3] = e.BOOST; o[4] = e.BPH; o[5] = this.dec === 1 ? 1 : 0; o[6] = 9;
      for (i = 0; i < 16; i++) { o[8 + i] = e.ee[i]; o[24 + i] = e.abE[i]; o[40 + i] = e.flM[i]; }
      return true;
    };
    BrmSynthV9.prototype.cloneWithListener = function (listener) {
      var MD = this.MD, P = MD.P, PH = MD.PH, GE = MD.GE, QD = MD.QD, FIRE = MD.FIRE, BANK = MD.BANK, TWW = MD.TWW, CDN = MD.CDN, POS = MD.POS, stubXyz = MD.stubXyz, perspective = MD.perspective;
      var c = new BrmSynthV9(MD, this.spec, listener, this.fs, { quality: this.quality(), seed: this.seed });
      c.auto = this.auto; c.eng.copyKernel(this.eng); c.speed = this.speed; c.cpuMs = this.cpuMs; c.cpuAudio = this.cpuAudio;
      return c;
    };
    BrmSynthV9.prototype._internal = function (m, rpm, thr, eng, L, R) {
      var MD = this.MD, P = MD.P, PH = MD.PH, GE = MD.GE, QD = MD.QD, FIRE = MD.FIRE, BANK = MD.BANK, TWW = MD.TWW, CDN = MD.CDN, POS = MD.POS, stubXyz = MD.stubXyz, perspective = MD.perspective;
      for (var k = 0; k < m; k++) { L[k] = 0; R[k] = 0; }
      eng.run(m, rpm, thr, L, R, this.gain);
    };
    /** n output samples (n <= 128) */
    BrmSynthV9.prototype.render = function (n, rpm, thr, L, R) {
      var MD = this.MD, P = MD.P, PH = MD.PH, GE = MD.GE, QD = MD.QD, FIRE = MD.FIRE, BANK = MD.BANK, TWW = MD.TWW, CDN = MD.CDN, POS = MD.POS, stubXyz = MD.stubXyz, perspective = MD.perspective;
      if (rpm < 60) rpm = 60;
      if (thr < 0) thr = 0; if (thr > 1) thr = 1;
      var t0 = this.auto && this.dec === 1 ? Date.now() : 0, k;
      if (this.dec === 1) {
        if (this.bL.length < n) { this.bL = new Float64Array(n); this.bR = new Float64Array(n); }
        this._internal(n, rpm, thr, this.eng, this.bL, this.bR);
        for (k = 0; k < n; k++) { L[k] = this.bL[k]; R[k] = this.bR[k]; }
      } else {
        // half rate: produce enough internal samples to fill n outputs (FIFO keeps the odd one)
        var need = n - this.fn, m = (need + 1) >> 1;
        if (m > 0) {
          this._internal(m, rpm, thr, this.eng, this.bL, this.bR);
          for (k = 0; k < m; k++) {
            this.up[0].push(this.bL[k], this.fifoL, this.fn); this.up[1].push(this.bR[k], this.fifoR, this.fn); this.fn += 2;
          }
        }
        for (k = 0; k < n; k++) { L[k] = this.fifoL[k]; R[k] = this.fifoR[k]; }
        for (k = n; k < this.fn; k++) { this.fifoL[k - n] = this.fifoL[k]; this.fifoR[k - n] = this.fifoR[k]; }
        this.fn -= n;
      }
      if (this.xf) this._crossfade(n, rpm, thr, L, R);
      if (t0) {
        this.cpuMs += Date.now() - t0; this.cpuAudio += n / this.fs * 1000;
        if (this.cpuAudio >= 3000) {
          if (this.cpuMs > 0.45 * this.cpuAudio) this._toLite();
          this.cpuMs = 0; this.cpuAudio = 0;
        }
      }
    };
    BrmSynthV9.prototype._toLite = function () {   // CPU guard: rebuild at half rate and crossfade (50 ms)
      var MD = this.MD, P = MD.P, PH = MD.PH, GE = MD.GE, QD = MD.QD, FIRE = MD.FIRE, BANK = MD.BANK, TWW = MD.TWW, CDN = MD.CDN, POS = MD.POS, stubXyz = MD.stubXyz, perspective = MD.perspective;
      var old = this.eng, ne = new Eng(MD, this.fs / 2, old.pp, this.seed);
      ne.TH = old.TH; ne.THR = old.THR; ne.BOOST = old.BOOST; ne.TSTUB = old.TSTUB; ne.EM = old.EM; ne.BPH = old.BPH; ne._r = old._r; ne._t = old._t; ne.speed = old.speed;
      this.xf = { eng: old, a: 1, bL: new Float64Array(128), bR: new Float64Array(128) };
      this.eng = ne; this.dec = 2; this.auto = false; this.fn = 0;
    };
    BrmSynthV9.prototype._crossfade = function (n, rpm, thr, L, R) {
      var MD = this.MD, P = MD.P, PH = MD.PH, GE = MD.GE, QD = MD.QD, FIRE = MD.FIRE, BANK = MD.BANK, TWW = MD.TWW, CDN = MD.CDN, POS = MD.POS, stubXyz = MD.stubXyz, perspective = MD.perspective;
      var x = this.xf, k, step = 1 / (0.05 * this.fs);
      this._internal(n, rpm, thr, x.eng, x.bL, x.bR);
      for (k = 0; k < n; k++) { x.a = Math.max(0, x.a - step); var b = 1 - x.a; L[k] = b * L[k] + x.a * x.bL[k]; R[k] = b * R[k] + x.a * x.bR[k]; }
      if (x.a <= 0) this.xf = null;
    };
    /** one BRM v9.x sound model = the shared kernel + one fitted parameter set */
    function build(id, setKey, label) {
    var P = Object.assign({ phys: SETS[setKey].phys, geom: SETS[setKey].geom, q: SETS[setKey].q, set: setKey }, COMMON);
    var PH = P.phys, GE = P.geom, QD = P.q;
    // ---------------------------------------------------------------- geometry (brm9.py)
    var FIRE = new Float64Array(16), BANK = new Int32Array(16), TWW = new Float64Array(16), CDN = new Float64Array(16), POS = new Int32Array(16);
    (function () {
      var DIST = [4, 3, 2, 1, 1, 2, 3, 4], TW = { 1: 0.4, 2: 0.7, 3: 0.9, 4: 1.0 }, c, i;
      for (var k = 0; k < 16; k++) FIRE[P.fire_order[k] - 1] = 45.0 * k;
      for (c = 1; c <= 16; c++) {
        i = (c - 1) % 8; BANK[c - 1] = c <= 8 ? 0 : 1; TWW[c - 1] = TW[DIST[i]]; CDN[c - 1] = (DIST[i] - 2.5) / 3.0;
        POS[c - 1] = (c > 8 && GE.b_mirror) ? 7 - i : i;
      }
    })();
    function stubXyz(c) {
      var p = PH.exit_pitch, i = (c - 1) % 8, gap = 0.05 * p / 0.06;
      if (c > 8 && GE.b_mirror) i = 7 - i;
      var x = p * (3.5 - i) + (i < 4 ? 0.5 * gap : -0.5 * gap);
      if (c > 8) x += PH.bank_dx;
      return [x, c <= 8 ? GE.stub_y : -GE.stub_y, GE.stub_z];
    }
    /** listening perspective for a game position: 'rear' = Python PERSPECTIVES.rear (chase view, static), in_cabin = onboard (fitted mic) */
    function perspective(pos) {
      if (pos === 'in_cabin' || pos === 'onboard') {
        var x = PH.onb_x, y = PH.onb_y, z = PH.onb_z, o = P.persp.onboard;
        return { name: 'onboard', ears: [[x, y + 0.09, z], [x, y - 0.09, z]], ground: PH.onb_ground > 0 ? PH.onb_ground : o.ground, shadow_far_db: o.shadow_far_db, wind: o.wind, outside: false };
      }
      var q = P.persp[pos === 'side' ? 'side' : 'rear'];
      return { name: pos === 'side' ? 'side' : 'rear', ears: q.ears, ground: q.ground, shadow_far_db: q.shadow_far_db, wind: q.wind, outside: true, ambience: q.ambience };
    }
    var MD = { P: P, PH: PH, GE: GE, QD: QD, FIRE: FIRE, BANK: BANK, TWW: TWW, CDN: CDN, POS: POS, stubXyz: stubXyz, perspective: perspective };
    function Synth(spec, listener, fs, opts) { return new BrmSynthV9(MD, spec, listener, fs, opts); }
    function EngM(fs, pp, seed) { return new Eng(MD, fs, pp, seed); }
    return { id: id, label: label, set: setKey, MD: MD, micG: MICG, Synth: Synth, synthOpts: {}, positions: ['rear', 'in_cabin'], P: P, Eng: EngM, butter: butter, perspective: perspective, Rng: Rng, rngFor: rngFor };
    }
    return [build('v9', 'v9_6', 'v9.6'), build('v9_1', 'v9_1', 'v9.1'), build('v9_2', 'v9_2', 'v9.2'), build('v9_3', 'v9_3', 'v9.3'), build('v9_4', 'v9_4', 'v9.4'),
      build('v9_8', 'v9_8', 'v9.8'), build('v9_9', 'v9_9', 'v9.9'), build('v9_10', 'v9_10', 'v9.10'),
      build('v9_11', 'v9_11', 'v9.11')];
  },
function engineModuleV8p(core) {
    // ---- model data (tools/export_v8p_params.py: engine-sound v8phys.CARS + PHYS merged with fitted_params_v8phys.json)
    var DATA = /*V8P_DATA*/{"gt350":{"car":{"name":"1967 Shelby GT350 (289 Hi-Po)","bore":0.1016,"stroke":0.072898,"rod":0.130937,"cr":10.5,"fire_order":[1,5,4,2,6,3,7,8],"left_bank":[5,6,7,8],"adv_dur":306.0,"lsa":114.0,"ev_d":0.04064,"ev_lift":0.0116078,"camx":0.9,"lifters":"solid","idle_rpm":850.0,"redline":6800.0,"inertia":0.2,"carb":"1x4 Holley 715 cfm","th_area_wot":0.005771714879094934,"v_man":0.0026,"v_ac":0.006,"sn_len":0.05,"sn_d":0.1,"th_len":0.06,"exhaust":"tri-y","prim_len":0.55,"prim_d":0.038099999999999995,"sec_len":0.35,"sec_d":0.047625,"head_len":0.5,"pipe_d":0.05715,"x_len":0.3,"x_d":0.0508,"inter_len":1.2,"muf":[["chamber",0.45,6.0,0.35]],"tail_len":0.85,"bank_dl":0.12,"tri_pairs":[[1,2],[3,4],[5,7],[6,8]],"mouth":[[-0.05,0.3,0.3],[-0.05,-0.3,0.3]],"intake":[3.75,0.0,0.85],"covers":[[3.55,0.32,0.75],[3.55,-0.32,0.75]],"starter_pos":[3.2,-0.3,0.35],"head":[1.75,0.35,1.0],"cabin_len":1.9,"cabin_h":1.05,"st_ratio":15.7,"st_teeth":157,"st_free_rpm":4800.0,"st_i_free":80.0,"st_i_lock":550.0,"st_v_test":10.6,"r_batt":0.013,"st_kind":"ford_pe","v_batt":12.4,"st_i_sat":200.0,"spark0":12.0},"phys":{"q_comb":4.4528726927691595,"heat_loss":0.13964927768935703,"n_poly":1.32,"burn":61.12339112063403,"xr0":0.06,"xr_ov":0.03468261820807481,"cov":0.12291346956848595,"cov_res":0.9118891425544978,"mis_res":0.25,"cd":0.65,"eta_v":0.85,"wall":0.002,"hf":0.00013531634473443194,"ro_loss":0.98,"t_prim0":750.0,"t_prima":420.0,"k_int":1.4578821185363917,"r_th":0.6,"k_tick":1.3099889610031585,"k_jet":0.33401186308125413,"f_floor":15.0,"f_fw":72.41426333042916,"k_cab":0.35,"cab_q":4.312931693945894,"cab_model":1,"leak_floor":0.3,"leak_fw":0.003,"k_floor":1.000048686830272,"k_ind":0.029701259811889123,"f_abs":8000.0,"k_win":0.3,"fric0":1.347449491470554,"fric1":0.12,"fric2":0.018,"fric_cold":2.14,"a_idle":7.285542239127752e-05,"fast_idle":1.0,"prime_rev":0.6,"catch_x":2.4,"react":0.3,"k_st":1.0,"k_hiss":24.59325797938332,"pack":0.06625559824334058,"bay_exh":0.8421933433992767,"k_nl":0.971139841482906,"dw_tau":0.4434368437152191,"gain":1.0}},"cadillac59":{"car":{"name":"1959 Cadillac Series 62 (390)","bore":0.1016,"stroke":0.098425,"rod":0.16827499999999998,"cr":10.5,"fire_order":[1,8,4,3,6,5,7,2],"left_bank":[1,3,5,7],"adv_dur":264.0,"lsa":114.0,"ev_d":0.039624,"ev_lift":0.01016,"camx":1.1,"lifters":"hydraulic","idle_rpm":520.0,"redline":5000.0,"inertia":0.45,"carb":"1x4 Carter AFB","th_area_wot":0.004979984443005094,"v_man":0.0032,"v_ac":0.009,"sn_len":0.12,"sn_d":0.075,"th_len":0.06,"exhaust":"log","port_len":0.12,"port_d":0.044449999999999996,"log_seg":0.11,"log_d":0.0508,"head_len":0.6,"pipe_d":0.0508,"x_len":0.25,"x_d":0.0508,"inter_len":1.5,"muf":[["chamber",0.25,8.0,0.0],["pipe",0.3,0.038099999999999995,0.0],["chamber",0.25,8.0,0.0],["pipe",0.35,0.0508,0.0],["chamber",0.35,5.0,0.1]],"tail_len":0.6,"bank_dl":0.25,"mouth":[[-0.1,0.55,0.45],[-0.1,-0.55,0.45]],"intake":[4.55,0.0,0.95],"covers":[[4.35,0.33,0.8],[4.35,-0.33,0.8]],"starter_pos":[3.95,0.3,0.35],"head":[2.3,0.4,1.05],"cabin_len":2.3,"cabin_h":1.05,"st_ratio":15.1,"st_teeth":166,"st_free_rpm":4350.0,"st_i_free":82.0,"st_i_lock":330.0,"st_v_test":10.6,"r_batt":0.012,"st_kind":"delco","v_batt":12.4,"st_i_sat":200.0,"spark0":5.0},"phys":{"q_comb":4.6,"heat_loss":0.08922799694585837,"n_poly":1.32,"burn":41.53321036001948,"xr0":0.06,"xr_ov":0.02,"cov":0.05032318942436391,"cov_res":0.4177863300939433,"mis_res":0.25,"cd":0.65,"eta_v":0.85,"wall":0.000965438109878401,"hf":0.0002345835862811804,"ro_loss":0.98,"t_prim0":731.0371827564371,"t_prima":420.0,"k_int":2.0,"r_th":0.6,"k_tick":1.1587881555704223,"k_jet":0.005,"f_floor":15.0,"f_fw":400.0,"k_cab":0.35,"cab_q":1.5,"cab_model":1,"leak_floor":0.3,"leak_fw":0.00529734107537258,"k_floor":1.003419309652263,"k_ind":0.03265282579623439,"f_abs":8000.0,"k_win":0.3,"fric0":1.8479475605609046,"fric1":0.12,"fric2":0.018,"fric_cold":1.2,"a_idle":5.1887408207057674e-05,"fast_idle":1.0,"prime_rev":0.7443756334908427,"catch_x":2.4,"react":0.3,"k_st":1.0,"k_hiss":6.541746651658922,"pack":1.7434902998281796,"bay_exh":0.4409732071662665,"k_nl":0.7884990991797303,"dw_tau":0.8054969365610245,"gain":1.0}},"p997":{"car":{"name":"2006 Porsche 911 Carrera 4S (997.1, M97.01 3.8)","bore":0.099,"stroke":0.0828,"rod":0.127,"cr":11.8,"fire_order":[1,6,2,4,3,5],"left_bank":[4,5,6],"adv_dur":252.0,"lsa":116.0,"ev_d":0.048648946545634474,"ev_lift":0.01,"camx":1.0,"lifters":"hydraulic","idle_rpm":670.0,"redline":7300.0,"inertia":0.16,"carb":"EFI, 1 electronic throttle body, two-stage resonance manifold","th_area_wot":0.004536459791783661,"v_man":0.0045,"v_ac":0.008,"sn_len":0.25,"sn_d":0.09,"th_len":0.1,"exhaust":"3-1","prim_len":0.3,"prim_x":[1.0,1.08,1.16],"prim_d":0.042,"head_len":0.25,"pipe_d":0.057,"x_len":0.0,"inter_len":0.6,"muf":[["chamber",0.3,4.0,0.15],["pipe",0.2,0.05,0.0],["chamber",0.22,10.0,0.5],["pipe",0.12,0.045,0.0],["chamber",0.22,10.0,0.5]],"tail_len":0.12,"bank_dl":0.04,"cross_mouth":true,"mouth":[[-0.02,0.52,0.3],[-0.02,-0.52,0.3]],"intake":[0.55,0.0,0.95],"covers":[[0.75,0.42,0.42],[0.75,-0.42,0.42]],"starter_pos":[1.3,0.22,0.35],"head":[2.15,0.35,1.0],"cabin_len":1.9,"cabin_h":1.0,"rear_engine":true,"eng_shadow":0.8,"st_ratio":54.888888888888886,"st_teeth":130,"st_free_rpm":16000.0,"st_i_free":70.0,"st_i_lock":850.0,"st_v_test":11.5,"st_v_lock":6.0,"r_batt":0.008,"st_kind":"bosch_pm","v_batt":12.5,"st_i_sat":10000.0,"spark0":10.0},"phys":{"q_comb":4.6,"heat_loss":0.06610091648405512,"n_poly":1.32,"burn":59.99578391306203,"xr0":0.05,"xr_ov":0.02,"cov":0.06304950659730628,"cov_res":0.07350737085237899,"mis_res":0.25,"cd":0.65,"eta_v":0.85,"wall":0.0004324461745975085,"hf":0.0007752814319838385,"ro_loss":0.98,"t_prim0":737.582808083124,"t_prima":420.0,"k_int":2.0,"r_th":0.6,"k_tick":0.0,"k_jet":0.36735134466175745,"f_floor":15.0,"f_fw":374.7941575801626,"k_cab":0.35,"cab_q":2.510320898642034,"cab_model":1,"leak_floor":0.3,"leak_fw":0.0035613790028847887,"k_floor":0.9663133290728505,"k_ind":0.5,"f_abs":8000.0,"k_win":0.3,"fric0":1.0197330664957027,"fric1":0.12,"fric2":0.018,"fric_cold":1.8,"a_idle":3.5e-05,"fast_idle":1.0,"prime_rev":2.0,"catch_x":2.4,"react":0.3,"k_st":1.0,"k_hiss":40.0,"pack":2.212846561320972,"bay_exh":0.17202969592712633,"k_nl":0.9301919446001485,"dw_tau":0.10000000000000002,"gain":1.0,"choke_tau":6.0}}}/*END_V8P_DATA*/;
    var P_ATM = 101325.0, R_GAS = 287.0, G_EX = 1.30, G_AIR = 1.40, T_AMB = 293.0, C_AIR = 343.0, RHO_AIR = 1.2;
    var DL = 1024, DLM = DL - 1, HB = 4096, NSRC = 6;
    // Q
    var Q_BORE_A = 0, Q_CRANK_R = 1, Q_ROD = 2, Q_VC = 3, Q_IVO = 4, Q_IVC = 5, Q_EVO = 6, Q_EVC = 7, Q_DV = 8, Q_LMAX = 9, Q_CAMX = 10,
      Q_CD = 11, Q_QC = 12, Q_HL = 13, Q_NP = 14, Q_BURN = 15, Q_XR0 = 16, Q_XROV = 17, Q_OVL = 18, Q_COV = 19, Q_COVR = 20, Q_MISR = 21,
      Q_ETAV = 22, Q_VD = 23, Q_VMAN = 24, Q_ATHW = 25, Q_AIDLE = 26, Q_FIDLE = 27, Q_ROL = 28, Q_J = 29, Q_F0 = 30, Q_F1 = 31, Q_F2 = 32,
      Q_FCOLD = 33, Q_STG = 34, Q_STKE = 35, Q_STRM = 36, Q_STRB = 37, Q_STIF = 38, Q_STJM = 39, Q_TEETH = 40, Q_VBAT = 41, Q_SOLID = 42,
      Q_KJET = 43, Q_VAC = 44, Q_LTH = 45, Q_RTH = 46, Q_SNM = 47, Q_SNR = 48, Q_SPARK0 = 49, Q_TAILA = 50, Q_NSEG = 51, Q_DWT = 52, Q_KNL = 53, Q_STIS = 54, Q_INSCALE = 55, NQ = 56;
    var S_TH = 0, S_W = 1, S_DW = 2, S_TM = 3, S_PMAN = 4, S_WI = 5, S_PPL = 6, S_QTH = 7, S_PAC = 8, S_QSN = 9, S_WM = 10, S_LOCK = 11,
      S_TGAS = 12, S_MESH = 13, S_COMM = 14, S_RNG = 15, S_TPRIM = 16, S_TTAIL = 17, S_NFIRE = 18, S_NEVT = 19, S_REVS = 20, S_JET0 = 21,
      S_JET1 = 22, S_TST = 23, S_CUR = 24, S_LOAD = 25, S_TMD = 26, S_NT = 27, S_DWM = 28, NS = 32;
    var C_PHI = 0, C_M = 1, C_K = 2, C_OPEN = 3, C_PIVC = 4, C_VIVC = 5, C_TIVC = 6, C_RP = 7, C_VALID = 8, C_P = 9, C_Q = 10, C_E = 11,
      C_XRES = 12, C_ADV = 13, C_DUR = 14, C_OFF = 15, C_BANK = 16, C_PIPE = 17, NC = 20;
    var F_D = 0, F_A = 1, F_GAIN = 2, F_LP = 3, F_SA = 4, F_SB = 5, F_C = 6, F_RP = 7, NF = 8;
    var K_THR0 = 0, K_THR1 = 1, K_RPM0 = 2, K_RPM1 = 3, K_MODE = 4, K_IGN = 5, K_SOL = 6, K_PIN = 7, K_FIREP = 8, K_CHOKE = 9, K_COLD = 10, K_BLEND = 11, NK = 12;
    var O_U0 = 0, O_U1 = 1, O_INT = 2, O_TICK0 = 3, O_TICK1 = 4, O_ST = 5, O_JET0 = 6, O_JET1 = 7, O_RPM = 8, O_HISS = 9, NO = 10;

    function rnd(S) {
      var x = S[S_RNG] >>> 0;
      x = (x ^ (x << 13)) >>> 0; x = (x ^ (x >>> 17)) >>> 0; x = (x ^ (x << 5)) >>> 0;
      S[S_RNG] = x; return x / 4294967296.0;
    }
    function vol(Q, phi) {
      var a = phi * Math.PI / 180.0, r = Q[Q_CRANK_R], l = Q[Q_ROD], s = Math.sin(a);
      return Q[Q_VC] + Q[Q_BORE_A] * (r * (1.0 - Math.cos(a)) + l - Math.sqrt(l * l - r * r * s * s));
    }
    function dvdth(Q, phi) {
      var a = phi * Math.PI / 180.0, r = Q[Q_CRANK_R], l = Q[Q_ROD], s = Math.sin(a), c = Math.cos(a);
      return Q[Q_BORE_A] * r * s * (1.0 + r * c / Math.sqrt(l * l - r * r * s * s));
    }
    function psi(pr, g) {
      var pc = Math.pow(2.0 / (g + 1.0), g / (g - 1.0));
      if (pr < pc) pr = pc; if (pr > 1.0) pr = 1.0;
      return Math.sqrt(2.0 * g / (g - 1.0) * (Math.pow(pr, 2.0 / g) - Math.pow(pr, (g + 1.0) / g)));
    }
    function crossed(a, b, e) { if (b >= a) return a < e && e <= b; return e > a || e <= b; }
    function aeffOf(Q, phi) {
      var lf = (phi - Q[Q_EVO]) / (Q[Q_EVC] - Q[Q_EVO]);
      if (lf <= 0.0 || lf >= 1.0) return 0.0;
      var lift = Q[Q_LMAX] * Math.pow(Math.sin(Math.PI * lf), Q[Q_CAMX]), a = Math.PI * Q[Q_DV] * lift, amax = 0.85 * Math.PI * Q[Q_DV] * Q[Q_DV] / 4.0;
      return Q[Q_CD] * (a < amax ? a : amax);
    }
    function vq(qg, p, m, V, Kc, tc, tpr, aeff, base, z0, rho_e, fs, dt) {
      var g = G_EX, pport = base + z0 * qg, dm, lim;
      if (pport < 0.2 * P_ATM) pport = 0.2 * P_ATM;
      if (p >= pport) {
        dm = aeff * p / Math.sqrt(R_GAS * tc) * psi(pport / p, g) * dt;
        var meq = V * Math.pow(pport / Kc, 1.0 / g);
        lim = m > meq ? 0.5 * (m - meq) : 0.0;
        if (dm > lim) dm = lim;
      } else {
        dm = -aeff * pport / Math.sqrt(R_GAS * tpr) * psi(p / pport, g) * dt;
        lim = 0.5 * (pport - p) * V / (g * R_GAS * tpr);
        if (-dm > lim) dm = -lim;
      }
      return dm * fs / rho_e;
    }

    /** the per-sample kernel (v8phys._kv8). CY, PF, AB, BB, NA, NODE, out: flat typed arrays (row-major) */
    /** arrival time of the forward sample just written (finite-amplitude steepening, BRM v9.11; v8phys._arr) */
    function arr(AB, AT, PF, p, wi, nt, knl) {
      var pf = p * NF, c = PF[pf + F_C], beta = knl * 0.5 * (G_EX + 1.0), u = AB[p * DL + wi] * c / (G_EX * P_ATM);
      if (beta * u < -0.6 * c) u = -0.6 * c / beta;
      var ta = nt + PF[pf + F_D] * c / (c + beta * u);
      if (ta > nt + DLM - 2) ta = nt + DLM - 2;
      var tp = AT[p * DL + ((wi - 1) & DLM)];
      if (ta < tp + 0.02) ta = tp + 0.02;
      AT[p * DL + wi] = ta;
    }

    function kv8(n, fs, K, Q, S, CY, PF, NPI, NODE, NN, NW, AB, BB, AT, NA, out) {
      var dt = 1.0 / fs, rad = Math.PI / 180.0, g = G_EX, k, i, p, j, knl = Q[Q_KNL], NCY = (CY.length / NC) | 0;   // job 8: 8 (V8) or 6 (flat-six) cylinders
      for (k = 0; k < n; k++) {
        var u = (k + 1.0) / n, ko = k * NO;
        var thr = K[K_THR0] + (K[K_THR1] - K[K_THR0]) * u;
        var a_th = Q[Q_AIDLE] * (1.0 + Q[Q_FIDLE] * K[K_CHOKE]) + Q[Q_ATHW] * (1.0 - Math.cos(thr * Math.PI / 2.0));
        var w = S[S_W];
        var m_th = Q[Q_CD] * a_th * P_ATM / Math.sqrt(R_GAS * T_AMB) * psi(S[S_PMAN] / P_ATM, G_AIR);
        var m_en = Q[Q_ETAV] * S[S_PMAN] / (R_GAS * T_AMB) * Q[Q_VD] * (w > 0.0 ? w : 0.0) / (4.0 * Math.PI);
        S[S_PMAN] += R_GAS * T_AMB / Q[Q_VMAN] * (m_th - m_en) * dt;
        if (S[S_PMAN] < 0.12 * P_ATM) S[S_PMAN] = 0.12 * P_ATM;
        if (S[S_PMAN] > P_ATM) S[S_PMAN] = P_ATM;
        var pman = S[S_PMAN];
        // electric starter
        var G = Q[Q_STG], ke = Q[Q_STKE], Rm = Q[Q_STRM] + Q[Q_STRB];
        var powered = K[K_SOL] > 0.5, engaged = K[K_PIN] > 0.5, wm = S[S_WM];
        if (!engaged) S[S_LOCK] = 0.0;
        if (S[S_LOCK] > 0.5) wm = G * (w > 0.0 ? w : 0.0);
        // series-wound motor: k(I) = a I / (1 + I / I_s), k(I_free) = ke (see v8phys.py)
        var cur = 0.0, kI = ke;
        if (powered) {
          var Is = Q[Q_STIS], a_s = ke * (1.0 + Q[Q_STIF] / Is) / Q[Q_STIF], qa = Rm / Is, qb = Rm + a_s * wm - Q[Q_VBAT] / Is;
          cur = (-qb + Math.sqrt(qb * qb + 4.0 * qa * Q[Q_VBAT])) / (2.0 * qa); if (cur < 0.0) cur = 0.0;
          kI = a_s * cur / (1.0 + cur / Is);
        }
        var loss = Q[Q_STIF] * (wm > 5.0 ? 1.0 : wm / 5.0);
        var tm = powered ? kI * (cur - loss) : -ke * 0.3 * loss;
        var tst = 0.0;
        if (S[S_LOCK] > 0.5 && tm > 0.0) tst = G * tm;
        else {
          S[S_LOCK] = 0.0;
          wm += tm / Q[Q_STJM] * dt;
          if (wm < 0.0) wm = 0.0;
          if (engaged && powered && wm >= G * (w > 0.0 ? w : 0.0)) S[S_LOCK] = 1.0;
        }
        S[S_WM] = wm; S[S_TST] = tst; S[S_CUR] = cur;
        var jtot = Q[Q_J] + (S[S_LOCK] > 0.5 ? Q[Q_STJM] * G * G : 0.0);
        // crank
        var krpm = (w > 0.0 ? w : 0.0) * 60.0 / (2.0 * Math.PI) / 1000.0;
        var visc = 1.0 + (Q[Q_FCOLD] - 1.0) * K[K_COLD];
        var tf = (Q[Q_F0] + Q[Q_F1] * krpm + Q[Q_F2] * krpm * krpm) * visc * 1e5 * Q[Q_VD] / (4.0 * Math.PI);
        var tg = S[S_TGAS], wf;
        if (w > 0.0) wf = w + (tg + tst - tf) / jtot * dt;
        else if (tg + tst > tf) wf = (tg + tst - tf) / jtot * dt;
        else wf = 0.0;
        if (wf < 0.0) wf = 0.0;
        // running mean of the gas torque over about one engine cycle (both modes); see v8phys.py
        var tmk = krpm > 0.8 ? 120.0 / (krpm * 1000.0) : 0.15;
        if (tmk < 0.015) tmk = 0.015;
        var tm0 = S[S_TM];
        S[S_TM] += (tg - S[S_TM]) * (dt / tmk);
        S[S_TMD] += ((S[S_TM] - tm0) / dt - S[S_TMD]) * (dt / tmk);
        if (K[K_MODE] < 0.5) {
          var wt = (K[K_RPM0] + (K[K_RPM1] - K[K_RPM0]) * u) * 2.0 * Math.PI / 60.0;
          S[S_DW] += (tg - S[S_TM] - tmk * S[S_TMD]) / jtot * dt;
          S[S_DW] -= S[S_DW] * (dt / Q[Q_DWT]);
          S[S_DWM] += (S[S_DW] - S[S_DWM]) * (dt / Q[Q_DWT]);   // minus its own running mean (see v8phys.py)
          var dvm = 0.04 * wt, dv = S[S_DW] - S[S_DWM];               // bounded +-4 % (anti-windup), see v8phys.py
          if (dv > dvm) { S[S_DW] = S[S_DWM] + dvm; dv = dvm; } else if (dv < -dvm) { S[S_DW] = S[S_DWM] - dvm; dv = -dvm; }
          w = wf + (wt + dv - wf) * K[K_BLEND];
        } else { w = wf; S[S_DW] = 0.0; S[S_DWM] = 0.0; }
        S[S_W] = w;
        var dth = w * dt / rad, th = S[S_TH] + dth;
        if (th >= 720.0) th -= 720.0;
        S[S_TH] = th; S[S_REVS] += dth / 360.0;
        var rpm = w * 60.0 / (2.0 * Math.PI);
        // delay-line arrivals
        var wi = S[S_WI] | 0, nt = S[S_NT];
        for (p = 0; p < NPI; p++) {
          var pf = p * NF, D = PF[pf + F_D], i0 = D | 0, fr = D - i0, j0 = (wi - i0) & DLM, j1 = (wi - i0 - 1) & DLM, pb = p * DL;
          var xb = (1.0 - fr) * BB[pb + j0] + fr * BB[pb + j1], xa;
          if (knl > 0.0) {
            var rp = PF[pf + F_RP] | 0, nx;
            for (;;) { nx = (rp + 1) & DLM; if (nx === wi || AT[pb + nx] > nt) break; rp = nx; }
            PF[pf + F_RP] = rp;
            nx = (rp + 1) & DLM;
            var t0a = AT[pb + rp], t1a = AT[pb + nx];
            if (nx === wi || t1a <= t0a) xa = AB[pb + rp];
            else { var fr0 = (nt - t0a) / (t1a - t0a); if (fr0 < 0.0) fr0 = 0.0; else if (fr0 > 1.0) fr0 = 1.0; xa = AB[pb + rp] + fr0 * (AB[pb + nx] - AB[pb + rp]); }
          } else xa = (1.0 - fr) * AB[pb + j0] + fr * AB[pb + j1];
          var al = PF[pf + F_LP];
          PF[pf + F_SA] += (1.0 - al) * (PF[pf + F_GAIN] * xa - PF[pf + F_SA]);
          PF[pf + F_SB] += (1.0 - al) * (PF[pf + F_GAIN] * xb - PF[pf + F_SB]);
        }
        // cylinders
        tg = 0.0;
        var qin = 0.0, tick0 = 0.0, tick1 = 0.0, tpr = S[S_TPRIM], rho_e = P_ATM / (R_GAS * tpr);
        var npol = Q[Q_NP] - (Q[Q_NP] - 1.12) * Math.exp(-rpm / 250.0), leak = Math.exp(-dt / 3.0);
        var hk = dt / 0.03 * Math.exp(-rpm / 100.0);   // burnt-charge wall heat transfer at near-standstill (v8phys.py)
        for (i = 0; i < NCY; i++) {
          var c = i * NC, phi0 = CY[c + C_PHI], phi = th - CY[c + C_OFF];
          if (phi < 0.0) phi += 720.0;
          CY[c + C_PHI] = phi;
          var V = vol(Q, phi), dvd = dvdth(Q, phi), tk = 0.0, m, pcy, vr;
          if (crossed(phi0, phi, Q[Q_IVC])) {
            var piv = pman;
            CY[c + C_PIVC] = piv; CY[c + C_VIVC] = vol(Q, Q[Q_IVC]);
            var xres = Q[Q_XR0] * Math.pow(P_ATM / piv, 1.0 / g) + Q[Q_XROV] * Q[Q_OVL] / 100.0 * (P_ATM / piv - 1.0) * Math.sqrt(1000.0 / (rpm > 100.0 ? rpm : 100.0));
            if (xres > 0.6) xres = 0.6;
            CY[c + C_XRES] = xres; CY[c + C_TIVC] = T_AMB + 30.0 + 500.0 * xres;
            var r1 = rnd(S), r2 = rnd(S);
            var gs = (rnd(S) + rnd(S) + rnd(S) + rnd(S) - 2.0) * 1.7320508075688772;
            var pm = xres > 0.25 ? Q[Q_MISR] * (xres - 0.25) : 0.0, e = 0.0;
            if (K[K_IGN] > 0.5 && r1 < K[K_FIREP] && r2 >= pm) {
              e = 1.0 + (Q[Q_COV] + Q[Q_COVR] * xres) * gs; if (e < 0.0) e = 0.0;
              S[S_NFIRE] += 1.0;
            }
            S[S_NEVT] += 1.0;
            CY[c + C_E] = e;
            var rc = 400.0 / (rpm > 50.0 ? rpm : 50.0); if (rc > 1.0) rc = 1.0;
            CY[c + C_RP] = 1.0 - Q[Q_HL] * (0.5 + 0.5 * rc) + e * Q[Q_QC] * (1.0 - xres);
            CY[c + C_ADV] = Q[Q_SPARK0] + (rpm < 450.0 ? 0.0 : (rpm < 3050.0 ? rpm - 450.0 : 2600.0) / 2600.0 * 20.0) + 12.0 * (1.0 - piv / P_ATM);
            CY[c + C_DUR] = Q[Q_BURN] * (1.0 + 0.8 * xres);
            CY[c + C_VALID] = 1.0;
            tk += 0.6;
          }
          if (crossed(phi0, phi, Q[Q_EVO])) {
            var ve = vol(Q, Q[Q_EVO]), pe, te;
            if (CY[c + C_VALID] > 0.5) {
              vr = CY[c + C_VIVC] / ve;
              pe = CY[c + C_PIVC] * Math.pow(vr, npol) * CY[c + C_RP];
              te = CY[c + C_TIVC] * CY[c + C_RP] * Math.pow(vr, npol - 1.0);
            } else { pe = pman; te = T_AMB; }
            m = pe * ve / (R_GAS * te);
            CY[c + C_M] = m; CY[c + C_K] = pe / Math.pow(m / ve, g); CY[c + C_OPEN] = 1.0; CY[c + C_VALID] = 0.0;
            tk += 1.0;
          }
          if (crossed(phi0, phi, Q[Q_EVC])) { CY[c + C_OPEN] = 0.0; tk += 0.8; }
          if (crossed(phi0, phi, Q[Q_IVO])) tk += 0.4;
          if (tk > 0.0) {
            var amp = tk * (0.15 + rpm / 1000.0) * (Q[Q_SOLID] > 0.5 ? 1.0 : 0.35);
            if (CY[c + C_BANK] < 0.5) tick0 += amp; else tick1 += amp;
          }
          var pp = CY[c + C_PIPE] | 0, ppf = pp * NF, pin = PF[ppf + F_SB];
          var aeff = CY[c + C_OPEN] > 0.5 ? aeffOf(Q, phi) : 0.0, ap = PF[ppf + F_A];
          var rv = (ap - aeff) / (ap + aeff) * Q[Q_ROL], z0 = rho_e * PF[ppf + F_C] / ap, q = 0.0;
          if (CY[c + C_OPEN] > 0.5) {
            m = CY[c + C_M];
            pcy = CY[c + C_K] * Math.pow(m / V, g);
            var tc = pcy * V / (m * R_GAS), base = P_ATM + (1.0 + rv) * pin, Kc = CY[c + C_K];
            var q0 = vq(0.0, pcy, m, V, Kc, tc, tpr, aeff, base, z0, rho_e, fs, dt);
            if (q0 === 0.0) q = 0.0;
            else {
              var a_ = 0.0, fa = q0, b_ = q0, fb = vq(q0, pcy, m, V, Kc, tc, tpr, aeff, base, z0, rho_e, fs, dt) - q0, side = 0;
              q = b_;
              for (var it = 0; it < 8; it++) {
                if (fb === fa) break;
                q = b_ - fb * (b_ - a_) / (fb - fa);
                var fq = vq(q, pcy, m, V, Kc, tc, tpr, aeff, base, z0, rho_e, fs, dt) - q;
                if (fq * fb < 0.0) { a_ = b_; fa = fb; if (side === -1) fa *= 0.5; side = -1; }
                else { if (side === 1) fa *= 0.5; side = 1; }
                b_ = q; fb = fq;
                if (Math.abs(fq) < 1e-7) break;
              }
            }
            var dm = q * rho_e / fs;
            if (dm >= 0.0) { m -= dm; CY[c + C_M] = m; pcy = CY[c + C_K] * Math.pow(m / V, g); }
            else {
              m -= dm; pcy = pcy - g * R_GAS * tpr * dm / V; CY[c + C_M] = m; CY[c + C_K] = pcy / Math.pow(m / V, g);
            }
            q = dm * fs / rho_e;
          } else {
            var psi_ = phi >= Q[Q_IVC] ? phi - 720.0 : phi;
            if (CY[c + C_VALID] > 0.5 && psi_ < Q[Q_EVO]) {
              vr = CY[c + C_VIVC] / V;
              var z = (psi_ + CY[c + C_ADV]) / CY[c + C_DUR], xb2 = 0.0;
              if (z > 0.0) { var zz = z < 1.6 ? z : 1.6; xb2 = 1.0 - Math.exp(-5.0 * zz * zz * zz); }
              CY[c + C_PIVC] *= leak;
              if (xb2 > 0.0) CY[c + C_RP] += ((1.0 - Q[Q_HL]) - CY[c + C_RP]) * hk * xb2;
              pcy = CY[c + C_PIVC] * Math.pow(vr, npol) * (1.0 + (CY[c + C_RP] - 1.0) * xb2);
            } else pcy = pman;
          }
          if (phi >= Q[Q_IVO] && phi < Q[Q_IVC]) qin += dvd * w * Q[Q_INSCALE];
          CY[c + C_Q] = q; CY[c + C_P] = pcy;
          AB[pp * DL + wi] = z0 * q + rv * pin;
          if (knl > 0.0) arr(AB, AT, PF, pp, wi, nt, knl);
          tg += (pcy - P_ATM) * dvd;
        }
        S[S_TGAS] = tg;
        // exhaust network nodes
        for (var nd = 0; nd < NN; nd++) {
          var nb = nd * NW, typ = NODE[nb];
          if (typ === 0) {
            var npt = NODE[nb + 2], sy = 0.0, sp = 0.0, p_, pin_;
            for (j = 0; j < npt; j++) {
              p_ = NODE[nb + 3 + 2 * j];
              var y = PF[p_ * NF + F_A] * PF[p_ * NF + F_C];
              pin_ = NODE[nb + 4 + 2 * j] === 0 ? PF[p_ * NF + F_SB] : PF[p_ * NF + F_SA];
              sy += y; sp += y * pin_;
            }
            var pj = 2.0 * sp / sy;
            for (j = 0; j < npt; j++) {
              p_ = NODE[nb + 3 + 2 * j];
              if (NODE[nb + 4 + 2 * j] === 0) { AB[p_ * DL + wi] = pj - PF[p_ * NF + F_SB]; if (knl > 0.0) arr(AB, AT, PF, p_, wi, nt, knl); }
              else BB[p_ * DL + wi] = pj - PF[p_ * NF + F_SA];
            }
          } else if (typ === 2) {
            var mo = NODE[nb + 1], pq = NODE[nb + 3], pinm = PF[pq * NF + F_SA];
            var st = NA[nd * 2] + NA[nd * 2 + 1] * (pinm - NA[nd * 2]);
            NA[nd * 2] = st;
            BB[pq * DL + wi] = -st;
            var rho_t = P_ATM / (R_GAS * S[S_TTAIL]);
            var U = (pinm + st) * PF[pq * NF + F_A] / (rho_t * PF[pq * NF + F_C]);
            out[ko + O_U0 + mo] = U;
            var uu = U / Q[Q_TAILA]; if (uu < 0.0) uu = 0.0;
            var nz = (rnd(S) - 0.5) * 3.4641016151377544;
            var fj = uu / Math.sqrt(4.0 * Q[Q_TAILA] / Math.PI) + 150.0, aj = 1.0 - Math.exp(-2.0 * Math.PI * fj / fs);
            var r4 = (uu / 30.0) * (uu / 30.0);
            S[S_JET0 + mo] += aj * (nz * 7e-4 * Q[Q_KJET] * r4 * r4 - S[S_JET0 + mo]);
            out[ko + O_JET0 + mo] = S[S_JET0 + mo];
          } else if (typ === 3) {
            // closed / rigid end: pressure reflection +1, no radiation (Helmholtz cavity)
            var pc = NODE[nb + 3];
            BB[pc * DL + wi] = PF[pc * NF + F_SA];
          }
        }
        S[S_WI] = (wi + 1) & DLM;
        S[S_NT] = nt + 1.0;
        // intake
        var gp = G_AIR * P_ATM;
        S[S_PPL] += gp / Q[Q_VMAN] * (S[S_QTH] - qin) * dt;
        var ath = a_th > 2e-6 ? a_th : 2e-6, mth = RHO_AIR * Q[Q_LTH] / ath, rth = Q[Q_RTH] * RHO_AIR * C_AIR / ath;
        S[S_QTH] = (S[S_QTH] + (S[S_PAC] - S[S_PPL]) / mth * dt) / (1.0 + rth / mth * dt);
        S[S_PAC] += gp / Q[Q_VAC] * (S[S_QSN] - S[S_QTH]) * dt;
        S[S_QSN] = (S[S_QSN] - S[S_PAC] / Q[Q_SNM] * dt) / (1.0 + Q[Q_SNR] / Q[Q_SNM] * dt);
        out[ko + O_INT] = S[S_QSN];
        if (S[S_PMAN] < 0.528 * P_ATM) out[ko + O_HISS] = (rnd(S) - 0.5) * 3.4641016151377544 * m_th * Math.pow(1.0 - S[S_PMAN] / (0.528 * P_ATM), 0.5);
        out[ko + O_TICK0] = tick0; out[ko + O_TICK1] = tick1;
        var sst = 0.0;
        if (S[S_LOCK] > 0.5) {
          S[S_MESH] += Q[Q_TEETH] * w * dt / (2.0 * Math.PI);
          if (S[S_MESH] >= 1.0) { S[S_MESH] -= Math.floor(S[S_MESH]); sst += 0.3 + tst / (G * ke * Q[Q_VBAT] / Rm); }
        }
        if (wm > 1.0) {
          S[S_COMM] += Q[Q_NSEG] * wm * dt / (2.0 * Math.PI);
          if (S[S_COMM] >= 1.0) { S[S_COMM] -= Math.floor(S[S_COMM]); sst += 0.1 + cur / 400.0; }
        }
        out[ko + O_ST] += sst;
        out[ko + O_RPM] = rpm;
      }
    }

    function bq(x, c, ci, s, si) {
      var y = c[ci] * x + s[si];
      s[si] = c[ci + 1] * x - c[ci + 3] * y + s[si + 1];
      s[si + 1] = c[ci + 2] * x - c[ci + 4] * y;
      return y;
    }
    /** post: sources -> listener (v8phys._post). BQ flat (rows of 5), BS flat (rows of 2), PATH flat (rows of 5), y interleaved L/R */
    function post(n, fs, out, PV, PST, H, PATH, NPATH, BQ, BS, yL, yR, off) {
      var k, e, j, m;
      for (k = 0; k < n; k++) {
        var ko = k * NO, hi = PST[3] | 0;
        var U0 = out[ko + O_U0], U1 = out[ko + O_U1], qi = out[ko + O_INT];
        PST[4] += PV[7] * ((U0 - PST[0]) * fs - PST[4]);
        PST[5] += PV[7] * ((U1 - PST[1]) * fs - PST[5]);
        var x0 = PV[0] * PST[4] + out[ko + O_JET0], x1 = PV[0] * PST[5] + out[ko + O_JET1];
        var hs = out[ko + O_HISS];
        PST[6] += PV[9] * (hs - PST[6]);
        PST[7] += PV[10] * ((hs - PST[6]) - PST[7]);
        var x2 = PV[1] * PV[0] * (qi - PST[2]) * fs + PV[8] * PST[7];
        PST[0] = U0; PST[1] = U1; PST[2] = qi;
        var t0 = out[ko + O_TICK0] * PV[2], t1 = out[ko + O_TICK1] * PV[2];
        var x3 = bq(t0, BQ, 0, BS, 0) + 0.6 * bq(t0, BQ, 5, BS, 2);
        var x4 = bq(t1, BQ, 0, BS, 4) + 0.6 * bq(t1, BQ, 5, BS, 6);
        var sv = out[ko + O_ST] * PV[3];
        var x5 = bq(sv, BQ, 10, BS, 8) + 0.7 * bq(sv, BQ, 15, BS, 10);
        H[hi] = x0; H[HB + hi] = x1; H[2 * HB + hi] = x2; H[3 * HB + hi] = x3; H[4 * HB + hi] = x4; H[5 * HB + hi] = x5;
        for (e = 0; e < 2; e++) {
          var r0 = 0.0, r1 = 0.0, r2 = 0.0, r3 = 0.0;
          for (j = 0; j < NPATH; j++) {
            var pr = j * 5;
            if ((PATH[pr + 1] | 0) !== e) continue;
            var s = PATH[pr] | 0, D = PATH[pr + 3], i0 = D | 0, fr = D - i0, hb = s * HB;
            var v = (1.0 - fr) * H[hb + ((hi - i0) & (HB - 1))] + fr * H[hb + ((hi - i0 - 1) & (HB - 1))];
            var rt = PATH[pr + 4] | 0;
            if (rt === 0) r0 += PATH[pr + 2] * v; else if (rt === 1) r1 += PATH[pr + 2] * v; else if (rt === 2) r2 += PATH[pr + 2] * v; else r3 += PATH[pr + 2] * v;
          }
          var b = (6 + e * 16) * 2;
          if (PV[6] > 0.0) {
            var cab;
            if (PV[11] > 0.5) {   // cab_model 1 (v1.11.2): mass law + leak per route, induction aperture, trim absorption
              cab = PV[4] * bq(bq(r1, BQ, 20, BS, b) + bq(r2, BQ, 25, BS, b + 2) + bq(r3, BQ, 30, BS, b + 4), BQ, 35, BS, b + 6);
            } else {              // cab_model 0 (v1.11.1): intake rides the firewall route
              var c1 = bq(bq(r1, BQ, 20, BS, b), BQ, 25, BS, b + 2);
              var c2 = bq(bq(r2 + r3, BQ, 30, BS, b + 4), BQ, 35, BS, b + 6);
              cab = PV[4] * (c1 + c2);
            }
            var md = 0.0;
            for (m = 0; m < (PV[6] | 0); m++) md += bq(cab, BQ, 40 + 5 * m, BS, b + 8 + 2 * m);
            r0 += cab + md;
          }
          if (e === 0) yL[off + k] = PV[5] * r0; else yR[off + k] = PV[5] * r0;
        }
        PST[3] = (hi + 1) & (HB - 1);
      }
    }

    // k x one-pole low-pass (bilinear, corner f) + flat g as a biquad row (v8phys._lp1)
    function lp1(f, fs, g, k) {
      var K = Math.tan(Math.PI * Math.min(f, 0.45 * fs) / fs), b0 = K / (1 + K), a1 = (K - 1) / (K + 1);
      return [k * b0 + g, k * b0 + g * a1, 0.0, a1, 0.0];
    }
    // cab_model 1 rows 4..7: floor, firewall, induction, trim absorption (v8phys.cabin_rows)
    function cabinRows(P, fs) {
      return [lp1(P.f_floor, fs, P.k_floor * P.leak_floor, P.k_floor), lp1(P.f_fw, fs, P.leak_fw, 1.0),
        lp1(P.f_fw, fs, P.leak_fw + P.k_ind, 1.0), lp1(P.f_abs, fs, 0.0, 1.0)];
    }
    function rbj(kind, f, q, fs) {
      var w = 2 * Math.PI * f / fs, cw = Math.cos(w), sw = Math.sin(w), al = sw / (2 * q), b;
      if (kind === 'lp') b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]; else b = [al, 0.0, -al];
      var a = [1 + al, -2 * cw, 1 - al];
      return [b[0] / a[0], b[1] / a[0], b[2] / a[0], a[1] / a[0], a[2] / a[0]];
    }
    function dist(a, b) { var dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2]; return Math.sqrt(dx * dx + dy * dy + dz * dz); }
    function camEvents(c, intakeLsa, intakeDur) {
      var d = c.adv_dur / 2.0;
      var dIn = ((intakeDur == null) ? c.adv_dur : intakeDur) / 2.0;
      var lsaIn = (intakeLsa == null) ? c.lsa : intakeLsa;
      return [360.0 + (lsaIn - dIn), 360.0 + (lsaIn + dIn), 360.0 - (c.lsa + d), 360.0 - (c.lsa - d)];
    }
    /** Non-default 911 option. Official 997.1: intake phasing (40° crank vane range) + 3.6/11.0 mm lift switch, exhaust fixed, idle = small cam and low overlap. (3/10 mm is the 996 Turbo.)
     *  Flap closed under load ~3000-5500 rpm is 997.2-confirmed, probable for 997.1. Phasing map and closed-plenum scale are model rules. */
    function applyVariocamPlus(Q, c, rpm, thr, st) {
      st = st || {};
      var idle = c.idle_rpm, red = c.redline;
      rpm = Math.max(idle * 0.5, Math.min(red * 1.05, rpm));
      thr = Math.max(0, Math.min(1, thr));
      var high = !!st.high;
      if (high) { if (rpm < 2500 || thr < 0.18) high = false; }
      else { if (rpm > 3000 && thr > 0.28) high = true; }
      st.high = high;
      Q[Q_INSCALE] = high ? 1.0 : (3.6 / 11.0);
      var x = (rpm - idle) / Math.max(1, 4600 - idle);
      if (x < 0) x = 0; if (x > 1) x = 1;
      x *= 0.35 + 0.65 * Math.max(thr, 0.15);
      if (rpm > 6000) x *= Math.max(0.55, 1 - (rpm - 6000) / 2500);
      var adv = 40 * x;  // 997.1 VarioCam Plus vane range, crank degrees
      var ev = camEvents(c, c.lsa - adv);
      Q[Q_IVO] = ev[0]; Q[Q_IVC] = ev[1]; Q[Q_EVO] = ev[2]; Q[Q_EVC] = ev[3]; Q[Q_OVL] = ev[3] - ev[0];
      var closed = (rpm >= 3000 && rpm <= 5500 && thr >= 0.35);
      Q[Q_VMAN] = c.v_man * (closed ? 0.55 : 1);
      st.flap_closed = closed; st.advance = adv;
      return st;
    }
    function perspective(c, name) {
      var m = c.mouth[0], x, z;
      if (name === 'rear') { var mm = [m[0] - 1.0, m[1], 0.5]; return { ears: [[mm[0], mm[1] + 0.09, mm[2]], [mm[0], mm[1] - 0.09, mm[2]]], ground: 0.8, outside: true, shadow: 0.3 }; }
      if (name === 'rear_far') return { ears: [[-4.0, 0.09, 1.5], [-4.0, -0.09, 1.5]], ground: 0.8, outside: true, shadow: 0.3 };
      if (name === 'engine_bay') { x = c.intake[0]; z = c.intake[2]; return { ears: [[x - 0.25, 0.09, z + 0.55], [x - 0.25, -0.09, z + 0.55]], ground: 0.8, outside: true, shadow: 0.5, bay: true }; }
      var h = c.head; return { ears: [[h[0], h[1] + 0.09, h[2]], [h[0], h[1] - 0.09, h[2]]], ground: 0.0, outside: false, shadow: 1.0, window: name === 'in_cabin_open' };
    }
    /** exhaust network (v8phys.build_net / net_arrays) */
    function buildNet(c) {
      var pipes = [], nodes = [], left = {}, xp = -1;
      c.left_bank.forEach(function (x) { left[x] = 1; });
      function pipe(L, d, grp, pack) { pipes.push([L, d, Math.PI * d * d / 4, grp, pack || 0, 0]); return pipes.length - 1; }
      function node(t, x, ports) { nodes.push([t, x, ports]); }
      for (var bank = 0; bank < 2; bank++) {
        var cyls = [], i, head, prev, dl = bank === 1 ? c.bank_dl : 0.0;
        for (i = 1; i <= c.fire_order.length; i++) if ((left[i] === 1) === (bank === 0)) cyls.push(i);
        if (c.exhaust === 'tri-y') {
          var secs = [];
          c.tri_pairs.forEach(function (pr) {
            if (cyls.indexOf(pr[0]) < 0) return;
            var pa = pipe(c.prim_len, c.prim_d, 0), pb = pipe(c.prim_len * 1.04, c.prim_d, 0);
            node(1, pr[0] - 1, [[pa, 0]]); node(1, pr[1] - 1, [[pb, 0]]);
            var s = pipe(c.sec_len, c.sec_d, 0);
            node(0, 0, [[pa, 1], [pb, 1], [s, 0]]); secs.push(s);
          });
          head = pipe(c.head_len, c.pipe_d, 1);
          node(0, 0, [[secs[0], 1], [secs[1], 1], [head, 0]]);
        } else if (c.exhaust === '3-1') {   // job 8: flat-six, three primaries per bank into a 3-1 collector (997.1 S)
          var prims = [];
          for (var j3 = 0; j3 < cyls.length; j3++) {
            var p3 = pipe(c.prim_len * c.prim_x[j3], c.prim_d, 0);
            node(1, cyls[j3] - 1, [[p3, 0]]); prims.push([p3, 1]);
          }
          head = pipe(c.head_len, c.pipe_d, 1);
          node(0, 0, prims.concat([[head, 0]]));
        } else {
          prev = null;
          for (var jj = 0; jj < cyls.length; jj++) {
            var pp = pipe(c.port_len, c.port_d, 0);
            node(1, cyls[jj] - 1, [[pp, 0]]);
            var ports = [[pp, 1]];
            if (prev !== null) ports.push([prev, 1]);
            var nxt = jj < cyls.length - 1 ? pipe(c.log_seg, c.log_d, 0) : pipe(c.head_len, c.pipe_d, 1);
            ports.push([nxt, 0]); node(0, 0, ports); prev = nxt;
          }
          head = prev;
        }
        var hx = (c.x_len || 0) > 0;   // job 8: the 997.1 has no balance pipe, the banks stay separate
        if (bank === 0 && hx) xp = pipe(c.x_len, c.x_d, 1);
        var inter = pipe(c.inter_len + dl, c.pipe_d, 1);
        node(0, 0, hx ? [[head, 1], [xp, bank], [inter, 0]] : [[head, 1], [inter, 0]]);
        prev = inter;
        c.muf.forEach(function (mf) {
          if (mf[0] === 'cavity') {
            var Ln = mf.length > 4 ? mf[4] : 0.05, dn = mf.length > 5 ? mf[5] : 0.0381, Lm = mf.length > 6 ? mf[6] : 0.12;
            var main = pipe(Lm, c.pipe_d, 2, 0.0), neck = pipe(Ln, dn, 2, 0.0), cav = pipe(mf[1], mf[2], 2, mf[3]);
            node(0, 0, [[prev, 1], [main, 0], [neck, 0]]);
            node(0, 0, [[neck, 1], [cav, 0]]);
            node(3, 0, [[cav, 1]]);
            prev = main;
          } else {
            var d = mf[0] === 'chamber' ? c.pipe_d * Math.sqrt(mf[2]) : mf[2];
            var p = pipe(mf[1], d, 2, mf[3]); node(0, 0, [[prev, 1], [p, 0]]); prev = p;
          }
        });
        var tail = pipe(c.tail_len, c.pipe_d, 2);
        if (c.helmholtz) {
          var hz = c.helmholtz, neck = pipe(hz[0], hz[1], 2, 0.0), cav = pipe(hz[2], hz[3], 2, 0.1);
          node(0, 0, [[prev, 1], [tail, 0], [neck, 0]]);
          node(0, 0, [[neck, 1], [cav, 0]]);
          node(3, 0, [[cav, 1]]);
        } else {
          node(0, 0, [[prev, 1], [tail, 0]]);
        }
        node(2, c.cross_mouth ? 1 - bank : bank, [[tail, 1]]);   // 997.1: each bank's cat pipe crosses to the other side
      }
      var NW = 0; nodes.forEach(function (n) { NW = Math.max(NW, 3 + 2 * n[2].length); });
      var NODE = new Int32Array(nodes.length * NW);
      nodes.forEach(function (n, k) {
        NODE[k * NW] = n[0]; NODE[k * NW + 1] = n[1]; NODE[k * NW + 2] = n[2].length;
        n[2].forEach(function (pe, j) { NODE[k * NW + 3 + 2 * j] = pe[0]; NODE[k * NW + 4 + 2 * j] = pe[1]; });
      });
      return { PG: pipes, NODE: NODE, NN: nodes.length, NW: NW };
    }

    /** the synth: v8phys.V8Phys + Ctl. spec = { car_key, idle_rpm, redline, name }, listener = { position } */
    function V8pSynth(spec, listener, fs, opts) {
      opts = opts || {};
      var key = spec.car_key || (spec.engineKey === 'CADDY' ? 'cadillac59' : spec.engineKey === 'P997' ? 'p997' : 'gt350');   // v1.11.0: P997 = the 911's flat-six
      var D = DATA[key];
      if (!D) throw new Error('v8p: no data for ' + key);
      this.key = key; this.c = D.car; this.P = Object.assign({}, D.phys, opts.phys || {}); this.fs = fs;
      this.spec = { idle_rpm: this.c.idle_rpm, redline: this.c.redline, name: this.c.name, car_key: key };
      this.variocamPlus = key === 'p997' && !!(spec.variocam_plus || opts.variocam_plus);
      this._vc = {};
      this.selfGated = opts.mode !== 'run' && opts.mode !== 'raw';
      var N = buildNet(this.c);
      this.PG = N.PG; this.NODE = N.NODE; this.NN = N.NN; this.NW = N.NW; this.NP = N.PG.length;
      this.Q = this._q();
      this.S = new Float64Array(NS); this.NCY = this.c.fire_order.length; this.CY = new Float64Array(this.NCY * NC); this.PF = new Float64Array(this.NP * NF);
      this.AB = new Float64Array(this.NP * DL); this.BB = new Float64Array(this.NP * DL); this.AT = new Float64Array(this.NP * DL); this.NA = new Float64Array(this.NN * 2);
      var seed = opts.seed === undefined ? 1 : opts.seed;
      this.S[S_RNG] = (Math.imul(0x9E3779B9 | 0, seed + 1) >>> 0) || 1;
      this.S[S_PMAN] = P_ATM; this.S[S_TPRIM] = T_AMB; this.S[S_TTAIL] = T_AMB;
      var fo = this.c.fire_order, off = [], i, left = {};
      this.c.left_bank.forEach(function (x) { left[x] = 1; });
      for (i = 0; i < this.NCY; i++) off[fo[i] - 1] = 720.0 / this.NCY * i;   // even firing (V8 90 deg, flat-six 120 deg)
      for (i = 0; i < this.NCY; i++) { this.CY[i * NC + C_OFF] = off[i]; this.CY[i * NC + C_BANK] = left[i + 1] ? 0.0 : 1.0; }
      for (var nd = 0; nd < this.NN; nd++) if (this.NODE[nd * this.NW] === 1) this.CY[this.NODE[nd * this.NW + 1] * NC + C_PIPE] = this.NODE[nd * this.NW + 3];
      for (i = 0; i < this.NCY; i++) { var v0 = (0.0 - off[i]) % 720.0; this.CY[i * NC + C_PHI] = v0 < 0 ? v0 + 720.0 : v0; }
      this.pw = 0.0; this.load = 0.0;
      this.K = new Float64Array(NK);
      this.out = new Float64Array(256 * NO);
      this.ctlr = new Ctl(this, opts.cold === undefined ? true : !!opts.cold);
      this._pipes();
      this.listener = listener || { position: 'rear' };
      this._listener(this.listener.position || 'rear');
      this.game = { running: false, starter: false, warm: false, gear: 0, has: false };
      if (opts.mode === 'run' || opts.mode === 'raw') this.setRunning(this.c.idle_rpm);
    }
    V8pSynth.prototype._q = function () {
      var c = this.c, P = this.P, Q = new Float64Array(NQ);
      var A = Math.PI * c.bore * c.bore / 4, vd1 = A * c.stroke, ev = camEvents(c);
      Q[Q_BORE_A] = A; Q[Q_CRANK_R] = c.stroke / 2; Q[Q_ROD] = c.rod; Q[Q_VC] = vd1 / (c.cr - 1.0);
      Q[Q_IVO] = ev[0]; Q[Q_IVC] = ev[1]; Q[Q_EVO] = ev[2]; Q[Q_EVC] = ev[3]; Q[Q_OVL] = ev[3] - ev[0];
      Q[Q_DV] = c.ev_d; Q[Q_LMAX] = c.ev_lift; Q[Q_CAMX] = c.camx; Q[Q_CD] = P.cd;
      Q[Q_QC] = P.q_comb; Q[Q_HL] = P.heat_loss; Q[Q_NP] = P.n_poly; Q[Q_BURN] = P.burn;
      Q[Q_XR0] = P.xr0; Q[Q_XROV] = P.xr_ov; Q[Q_COV] = P.cov; Q[Q_COVR] = P.cov_res; Q[Q_MISR] = P.mis_res;
      Q[Q_ETAV] = P.eta_v; Q[Q_VD] = c.fire_order.length * vd1; Q[Q_VMAN] = c.v_man; Q[Q_ATHW] = c.th_area_wot;
      Q[Q_AIDLE] = P.a_idle > 0 ? P.a_idle : 2.0e-5; Q[Q_FIDLE] = P.fast_idle; Q[Q_ROL] = P.ro_loss;
      Q[Q_J] = c.inertia; Q[Q_F0] = P.fric0; Q[Q_F1] = P.fric1; Q[Q_F2] = P.fric2; Q[Q_FCOLD] = P.fric_cold;
      var vlock = c.st_v_lock !== undefined ? c.st_v_lock : c.st_kind === 'ford_pe' ? 5.0 : 3.5, Rm = vlock / c.st_i_lock, wfree = c.st_free_rpm * 2 * Math.PI / 60;
      Q[Q_STG] = c.st_ratio; Q[Q_STRM] = Rm; Q[Q_STRB] = c.r_batt;
      Q[Q_STKE] = (c.st_v_test - c.st_i_free * Rm) / wfree; Q[Q_STIF] = c.st_i_free; Q[Q_STJM] = 2.5e-4; Q[Q_STIS] = c.st_i_sat;
      Q[Q_TEETH] = c.st_teeth; Q[Q_VBAT] = c.v_batt; Q[Q_NSEG] = 22; Q[Q_SOLID] = c.lifters === 'solid' ? 1.0 : 0.0;
      Q[Q_KJET] = P.k_jet; Q[Q_VAC] = c.v_ac; Q[Q_LTH] = c.th_len; Q[Q_RTH] = P.r_th;
      var asn = Math.PI * c.sn_d * c.sn_d / 4;
      Q[Q_SNM] = RHO_AIR * (c.sn_len + 0.85 * c.sn_d) / asn; Q[Q_SNR] = 0.1 * RHO_AIR * C_AIR / asn;
      Q[Q_SPARK0] = c.spark0; Q[Q_TAILA] = Math.PI * c.pipe_d * c.pipe_d / 4; Q[Q_DWT] = P.dw_tau; Q[Q_KNL] = P.k_nl;
      Q[Q_INSCALE] = 1.0;
      return Q;
    };
    V8pSynth.prototype._pipes = function () {
      var P = this.P, fs = this.fs, S = this.S, tp = S[S_TPRIM], tt = S[S_TTAIL], tm = T_AMB + 0.65 * (tp - T_AMB), p, nd;
      var tail = {};
      for (nd = 0; nd < this.NN; nd++) if (this.NODE[nd * this.NW] === 2) tail[this.NODE[nd * this.NW + 3]] = 1;
      for (p = 0; p < this.NP; p++) {
        var g = this.PG[p], L = g[0], d = g[1], A = g[2], grp = g[3], pack = g[4];
        var T = grp === 0 ? tp : (grp === 1 ? tm : tt), c = Math.sqrt(G_EX * R_GAS * T);
        var Le = L + (tail[p] ? 0.61 * d / 2 : 0.0), f = p * NF;
        this.PF[f + F_C] = c; this.PF[f + F_A] = A;
        this.PF[f + F_D] = Math.min(Math.max(Le / c * fs, 1.0), DL - 4);
        this.PF[f + F_GAIN] = Math.exp(-P.wall * L / d - pack * (P.pack_r || 0));
        this.PF[f + F_LP] = Math.min(0.9, P.hf * L / d + pack * P.pack);
      }
      for (nd = 0; nd < this.NN; nd++) if (this.NODE[nd * this.NW] === 2) {
        var pq = this.NODE[nd * this.NW + 3], a = this.PG[pq][1] / 2, fc = this.PF[pq * NF + F_C] / (2 * Math.PI * a);
        this.NA[nd * 2 + 1] = 1.0 - Math.exp(-2 * Math.PI * Math.min(fc, 0.45 * fs) / fs);
      }
    };
    V8pSynth.prototype._listener = function (name, keepState) {
      var c = this.c, P = this.P, fs = this.fs, pr = perspective(c, name), rows = [], e, s;
      this.persp = pr; this.posName = name;
      var srcs = [c.mouth[0], c.mouth[1], c.intake, c.covers[0], c.covers[1], c.starter_pos];
      for (e = 0; e < 2; e++) for (s = 0; s < srcs.length; s++) {
        var ear = pr.ears[e], pos = srcs[s], front = s >= 2, r = dist(ear, pos);
        if (pr.outside) {
          var sh = 1.0;
          if (front && !pr.bay) sh = c.eng_shadow !== undefined ? c.eng_shadow : pr.shadow;   // job 8: rear engine behind the lid grille
          if (!front && pr.bay) sh = P.bay_exh;
          rows.push([s, e, sh / Math.max(r, 0.25), r / C_AIR * fs, 0]);
          var ri = dist(ear, [pos[0], pos[1], -pos[2]]);
          rows.push([s, e, sh * pr.ground / Math.max(ri, 0.25), ri / C_AIR * fs, 0]);
        } else {
          rows.push([s, e, 1.0 / Math.max(r, 0.25), r / C_AIR * fs, front ? (s === 2 ? 3 : 2) : 1]);
          if (pr.window && (P.cab_model || 0) >= 1) {   // open window: outside free field through the opening
            var sw = front ? P.k_win * (c.eng_shadow !== undefined ? c.eng_shadow : 0.3) : P.k_win;
            rows.push([s, e, sw / Math.max(r, 0.25), r / C_AIR * fs, 0]);
            var rw = dist(ear, [pos[0], pos[1], -pos[2]]);
            rows.push([s, e, 0.8 * sw / Math.max(rw, 0.25), rw / C_AIR * fs, 0]);
          }
        }
      }
      this.NPATH = rows.length; this.PATH = new Float64Array(rows.length * 5);
      rows.forEach(function (rw, j) { for (var t = 0; t < 5; t++) this.PATH[j * 5 + t] = rw[t]; }, this);
      var BQ = [rbj('bp', 2300.0, 10.0, fs), rbj('bp', 4300.0, 14.0, fs), rbj('bp', 950.0, 6.0, fs), rbj('bp', 2200.0, 8.0, fs),
        rbj('lp', P.f_floor, 0.707, fs), rbj('lp', P.f_floor, 0.707, fs), rbj('lp', P.f_fw, 0.707, fs), rbj('lp', P.f_fw, 0.707, fs)];
      if ((P.cab_model || 0) >= 1) { var cr = cabinRows(P, fs); for (var q = 0; q < 4; q++) BQ[4 + q] = cr[q]; }
      var modes = [C_AIR / (2 * c.cabin_len), C_AIR / (2 * c.cabin_h)];
      modes.forEach(function (f) { var b = rbj('bp', f, P.cab_q, fs), h = P.cab_q * 0.5; BQ.push([b[0] * h, b[1] * h, b[2] * h, b[3], b[4]]); });
      this.BQ = new Float64Array(BQ.length * 5);
      BQ.forEach(function (b, j) { for (var t = 0; t < 5; t++) this.BQ[j * 5 + t] = b[t]; }, this);
      if (!keepState || !this.BS) { this.BS = new Float64Array(80); this.H = new Float64Array(NSRC * HB); }
      if (!this.PST) this.PST = new Float64Array(8);
      var fka = C_AIR / (2 * Math.PI * c.pipe_d / 2);
      this.PV = new Float64Array([RHO_AIR / (4 * Math.PI), P.k_int, 0.02 * P.k_tick, 0.05 * P.k_st, P.k_cab, 0.25 * P.gain,
        pr.outside ? 0.0 : modes.length, 1.0 - Math.exp(-2 * Math.PI * fka / fs), P.k_hiss,
        1.0 - Math.exp(-2 * Math.PI * 1500.0 / fs), 1.0 - Math.exp(-2 * Math.PI * 7000.0 / fs), P.cab_model || 0]);
    };
    V8pSynth.prototype.cloneWithListener = function (lis) {
      var o = Object.create(V8pSynth.prototype); Object.assign(o, this);
      ['S', 'CY', 'PF', 'AB', 'BB', 'AT', 'NA', 'K', 'PST'].forEach(function (a) { o[a] = this[a].slice(); }, this);
      o.out = new Float64Array(256 * NO); o.game = Object.assign({}, this.game);
      o.ctlr = this.ctlr.clone(o);
      o.BS = null; o.listener = lis; o._listener((lis && lis.position) || 'rear');
      return o;
    };
    V8pSynth.prototype.setRunning = function (rpm) {
      this.S[S_W] = rpm * 2 * Math.PI / 60; this.S[S_PMAN] = 0.4 * P_ATM; this.pw = 1.0;
      this.S[S_TPRIM] = T_AMB + (this.P.t_prim0 - T_AMB); this.S[S_TTAIL] = T_AMB + 0.4 * (this.S[S_TPRIM] - T_AMB);
      this.ctlr.setRunning(); this._pipes();
    };
    V8pSynth.prototype._thermal = function (dtb, firing) {
      var P = this.P, S = this.S, pr = S[S_PMAN] / P_ATM, rpm = S[S_W] * 60 / (2 * Math.PI);
      var tl = firing ? pr * (0.5 + 0.5 * Math.min(1.0, rpm / 4000.0)) : 0.0;
      this.load += (tl - this.load) * Math.min(1.0, dtb / 1.5);
      if (firing) this.pw += (1.0 - this.pw) * dtb / 25.0; else this.pw -= this.pw * dtb / 200.0;
      var tg = firing ? (T_AMB + (P.t_prim0 + P.t_prima * this.load - T_AMB) * (0.6 + 0.4 * this.pw)) : T_AMB + 40.0 * this.pw;
      S[S_TPRIM] += (tg - S[S_TPRIM]) * Math.min(1.0, dtb / 0.4);
      var tt = T_AMB + (0.25 + 0.2 * this.pw) * (S[S_TPRIM] - T_AMB);
      S[S_TTAIL] += (tt - S[S_TTAIL]) * Math.min(1.0, dtb / 1.0);
    };
    /** one block (n <= 256) -> L/R at off; game = { running, starter, warm, gear } or null (offline 'run' mode) */
    V8pSynth.prototype.block = function (n, rpm, thr, game, L, R, off) {
      if (this.variocamPlus) this._vc = applyVariocamPlus(this.Q, this.c, rpm, thr, this._vc);
      var dtb = n / this.fs, K = this.ctlr.step(dtb, rpm, thr, game), out = this.out, i;
      for (i = 0; i < n * NO; i++) out[i] = 0.0;
      if (this.ctlr.click) { out[O_ST] += this.ctlr.click; this.ctlr.click = 0.0; }
      if (this.ctlr.silent) { for (i = 0; i < n; i++) { L[off + i] = 0; R[off + i] = 0; } return; }
      var nf0 = this.S[S_NFIRE];
      kv8(n, this.fs, K, this.Q, this.S, this.CY, this.PF, this.NP, this.NODE, this.NN, this.NW, this.AB, this.BB, this.AT, this.NA, out);
      var firing = K[K_IGN] > 0.5 && (this.S[S_NFIRE] > nf0 || this.ctlr.recentFire());
      this.ctlr.noteFire(this.S[S_NFIRE] - nf0);
      this._thermal(dtb, firing); this._pipes();
      post(n, this.fs, out, this.PV, this.PST, this.H, this.PATH, this.NPATH, this.BQ, this.BS, L, R, off);
    };
    // ---- EngineVoice / ListenerSource interface
    V8pSynth.prototype.render = function (n, rpm, thr, L, R, off) {
      off = off || 0;
      var gm = this.selfGated ? this.game : null;
      for (var s = 0; s < n;) { var m = Math.min(256, n - s); this.block(m, rpm, thr, gm, L, R, off + s); s += m; }
    };
    V8pSynth.prototype.ctl = function (m) {
      var g = this.game;
      if (m.running !== undefined) { g.running = !!m.running; g.has = true; }
      if (m.starter !== undefined) { g.starter = !!m.starter; g.has = true; }
      if (m.warm !== undefined) g.warm = !!m.warm;
      if (m.gear !== undefined) g.gear = m.gear;
    };
    /** the engine was already running when this source was created (model / car switch) */
    V8pSynth.prototype.initRunning = function (on) { if (on) { this.setRunning(this.c.idle_rpm); this.game.running = true; this.game.has = true; } };
    V8pSynth.prototype.report = function () { return this.ctlr.report(); };
    V8pSynth.prototype.rpmNow = function () { return this.S[S_W] * 60 / (2 * Math.PI); };

    function Ctl(syn, cold) {
      this.syn = syn; this.phase = 'off'; this.t = 0.0; this.key = false;
      this.t_key = 0.0; this.t_catch = null; this.t_rel = null; this.t_stop = null;
      this.handed = false; this.blend = 0.0; this.bt = 0.0; this.coldness = cold ? 1.0 : 0.0;
      this.revs0 = 0.0; this.nf0 = 0.0; this.wcrank = 0.0; this.ncrank = 0;
      this.sol = 0.0; this.pin = 0.0; this.click = 0.0; this.silent = true;
      this.fires = new Float64Array(16); this.fi = 0; this.idle = 0.0; this.lowt = 0.0; this.started_k = false;
    }
    Ctl.prototype.clone = function (syn) { var o = Object.create(Ctl.prototype); Object.assign(o, this); o.syn = syn; o.fires = this.fires.slice(); return o; };
    Ctl.prototype.setRunning = function () {
      this.phase = 'run'; this.handed = true; this.blend = 1.0; this.t_catch = -100.0; this.t_rel = -100.0;
      this.coldness = 0.0; this.sol = this.pin = 0.0; this.silent = false; this.key = true; this.fires.fill(1.0);
    };
    Ctl.prototype.noteFire = function (n) { this.fires[this.fi] = n; this.fi = (this.fi + 1) % 16; };
    Ctl.prototype.recentFire = function () { var s = 0; for (var i = 0; i < 16; i++) s += this.fires[i]; return s > 0; };
    Ctl.prototype.keyOn = function (cold) {
      if (cold !== null && cold !== undefined) this.coldness = cold ? 1.0 : 0.0;
      this.phase = 'crank'; this.t_key = this.t; this.t_catch = null; this.t_rel = null; this.t_stop = null;
      this.handed = false; this.blend = 0.0;
      this.revs0 = this.syn.S[S_REVS]; this.nf0 = this.syn.S[S_NFIRE]; this.wcrank = 0.0; this.ncrank = 0;
      this.silent = false; this.key = true;
      this.click = this.syn.c.st_kind === 'ford_pe' ? 1.2 : 0.0;
    };
    Ctl.prototype.keyOff = function () {
      this.key = false;
      if (this.phase === 'crank' || this.phase === 'run') {
        this.phase = 'shut'; this.t_stop = null;
        if (this.sol > 0.5) this.click += 1.5;
        this.sol = this.pin = 0.0;
      }
    };
    Ctl.prototype.step = function (dtb, rpm_in, thr_in, game) {
      var syn = this.syn, P = syn.P, S = syn.S, K = syn.K;
      this.t += dtb;
      var ford = syn.c.st_kind === 'ford_pe';
      if (game && game.has) {
        var on = !!(game.running || game.starter);
        if (on && !this.key && (this.phase === 'off' || this.phase === 'stopped' || this.phase === 'shut')) this.keyOn(!game.warm);
        else if (!on && this.key) this.keyOff();
      }
      var ph = this.phase, thr_prev = K[K_THR1], rpm_prev = this.started_k ? K[K_RPM1] : rpm_in;
      this.started_k = true;
      K.fill(0.0); K[K_THR0] = thr_prev;
      var w = S[S_W], rpm = w * 60 / (2 * Math.PI);
      if (ph === 'crank') {
        var tc = this.t - this.t_key;
        if (ford) { if (tc >= 0.03 && this.pin < 0.5) { this.pin = this.sol = 1.0; this.click += 1.6; } }
        else {
          if (tc >= 0.0 && this.pin < 0.5) { this.pin = 1.0; this.click += 1.4; }
          if (tc >= 0.035 && this.sol < 0.5) { this.sol = 1.0; this.click += 0.6; }
        }
        var revs = S[S_REVS] - this.revs0, pr = P.prime_rev * (this.coldness > 0.5 ? 1.0 : 0.3);
        K[K_FIREP] = 1.0 - Math.exp(-Math.max(0.0, revs - 0.3) / pr);
        if (tc > 6.0) K[K_FIREP] = 1.0;
        if (S[S_NFIRE] - this.nf0 < 1 && tc > 0.15) { this.ncrank += 1; this.wcrank += (rpm - this.wcrank) / this.ncrank; }
        if (S[S_NFIRE] - this.nf0 >= 3 && rpm > P.catch_x * Math.max(this.wcrank, 60.0)) { this.phase = 'run'; this.t_catch = this.t; }
        K[K_IGN] = 1.0; K[K_MODE] = 1.0; K[K_CHOKE] = this.coldness; K[K_COLD] = this.coldness;
      }
      if (this.phase === 'run') {
        var tcat = this.t - this.t_catch;
        if (this.t_rel === null && tcat >= P.react) { this.t_rel = this.t; if (this.sol > 0.5) this.click += 1.3; this.sol = this.pin = 0.0; }
        K[K_FIREP] = 1.0; K[K_IGN] = 1.0;
        var ch = Math.exp(-Math.max(tcat, 0.0) / (P.choke_tau > 0 ? P.choke_tau : 14.0));
        K[K_CHOKE] = this.coldness * ch; K[K_COLD] = this.coldness * ch;
        if (game && game.has) {
          var hold = this.coldness > 0.5 ? 1.6 : 1.0;
          if (!this.handed && (tcat > hold || (tcat > P.react + 0.1 && thr_in > 0.05))) { this.handed = true; this.bt = 0.0; }
          if (this.handed) { this.bt += dtb; this.blend = Math.min(1.0, this.bt / 0.35); }
        }
        if (this.handed) { K[K_MODE] = 0.0; K[K_BLEND] = this.blend; }
        else { K[K_MODE] = 1.0; this.idle = rpm; }
      }
      if (this.phase === 'shut') {
        K[K_IGN] = 0.0; K[K_MODE] = 1.0;
        if (game && game.gear && rpm_in > rpm) { K[K_MODE] = 0.0; K[K_BLEND] = 1.0; }
        if (rpm < 15.0) { this.lowt += dtb; if (this.lowt > 0.15) { this.phase = 'stopped'; S[S_W] = 0.0; this.t_stop = this.t; this.click += 0.5; } }
        else this.lowt = 0.0;
      }
      if (this.phase === 'stopped' && this.t - this.t_stop > 0.6) this.phase = 'off';
      if (this.phase === 'off') this.silent = true;
      K[K_SOL] = this.sol; K[K_PIN] = this.pin; K[K_THR1] = thr_in;
      K[K_RPM0] = rpm_prev; K[K_RPM1] = rpm_in;
      return K;
    };
    Ctl.prototype.report = function () {
      if (this.phase === 'off') return { phase: 'off', rpm: 0 };
      var r = this.syn.S[S_W] * 60 / (2 * Math.PI);
      return { phase: this.phase, rpm: r, idle: this.idle, handed: this.handed, cold: this.coldness > 0.5 };
    };

    function Synth(spec, listener, fs, opts) { return new V8pSynth(spec, listener, fs, opts); }
    return { id: 'v9p', label: 'v9.11 physics (V8)', Synth: Synth, synthOpts: { mode: 'game' }, startup: true, positions: ['rear', 'in_cabin'],
      V8pSynth: V8pSynth, DATA: DATA, perspective: perspective, buildNet: buildNet };
  }]);

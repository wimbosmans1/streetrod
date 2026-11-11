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
      this.model = model; this.fs = fs; this.listeners = listeners; this.views = views || ['rear', 'in_cabin', 'engine_bay'];
      this.pos = this.views[0];
      this.cur = new model.Synth(spec, listeners[this.pos], fs, opts); this.spec = this.cur.spec;
      this.g = this.gainOf(this.pos); this.next = null; this.ng = 1; this.xf = 1;
      this.aL = new Float32Array(128); this.aR = new Float32Array(128); this.bL = new Float32Array(128); this.bR = new Float32Array(128);
    }
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
      if (o.spec) this.msg({ type: 'spec', model: o.model || 'v1', spec: o.spec, listeners: o.listeners });
      if (o.script) { this.script = o.script; this.engGain = this.engGainTarget = 1; }
    }
    rnd() { this.seed = (Math.imul(this.seed, 1664525) + 1013904223) >>> 0; return this.seed / 4294967296 * 2 - 1; }
    makeSource(m) {
      const model = ENGINES[m.model || 'v1'];
      if (!model) throw new Error('unknown sound model ' + m.model);
      if (model.id === 'v1') return new V1Source(m.spec, this.fs);
      const s = new ListenerSource(model, m.spec, m.listeners || {}, m.views, this.fs, { fir: this.opts.fir || 'fft', ntap: this.opts.ntap });
      s.setView(this.view); if (s.next) { s.cur = s.next; s.pos = s.npos; s.g = s.ng; s.next = null; }
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
        if (m.rpm !== undefined) this.target.rpm = m.rpm;
        if (m.thr !== undefined) this.target.thr = m.thr;
        if (m.running !== undefined) this.engGainTarget = m.running ? 1 : 0;
        if (m.starter !== undefined) this.starterTarget = m.starter ? 1 : 0;
        if (m.crank !== undefined) this.crank = m.crank;
        if (m.speed !== undefined) this.speed = m.speed;
        if (m.skid !== undefined) this.skid = m.skid;
        if (m.listen !== undefined) {   // view: 0 outside rear, 1 inside, 2 engine bay
          this.view = m.listen | 0; this.listen = this.view === 1 ? 1 : 0;
          if (this.src) this.src.setView(this.view); if (this.nsrc) this.nsrc.setView(this.view);
        }
      } else if (m.type === 'script') { this.script = m.script; this.scriptPos = 0; }
    }
    process(L, R, n) {
      this.blocks++;
      if (!this.src) { for (let k = 0; k < n; k++) { L[k] = 0; R[k] = 0; } return; }
      let rpm, thr;
      if (this.script) {
        const i = Math.min(this.scriptPos, this.script.length - 1);
        rpm = this.script[i][0]; thr = this.script[i][1]; this.scriptPos++;
      } else {
        this.sm.rpm += (this.target.rpm - this.sm.rpm) * this.smK;
        this.sm.thr += (this.target.thr - this.sm.thr) * this.smK;
        rpm = Math.max(60, this.sm.rpm); thr = Math.min(1, Math.max(0, this.sm.thr));
      }
      const tL = this.tL, tR = this.tR, wv = this.wv;
      for (let k = 0; k < n; k++) { this.listenS += (this.listen - this.listenS) * this.listenK; wv[k] = this.listenS; }
      this.src.render(n, rpm, thr, tL, tR, wv, this.raw);
      if (this.raw) { for (let k = 0; k < n; k++) { L[k] = tL[k]; R[k] = tR[k]; } return; }
      for (let k = 0; k < n; k++) { tL[k] *= this.outGain; tR[k] *= this.outGain; }
      if (this.nsrc) {
        this.nsrc.render(n, rpm, thr, this.tL2, this.tR2, wv, false);
        const g2 = this.nsrc.outGain, step = 1 / (0.25 * this.fs);
        for (let k = 0; k < n; k++) {
          this.xfade = Math.min(1, this.xfade + step);
          const a = Math.cos(this.xfade * Math.PI / 2), b = Math.sin(this.xfade * Math.PI / 2);
          tL[k] = a * tL[k] + b * g2 * this.tL2[k]; tR[k] = a * tR[k] + b * g2 * this.tR2[k];
        }
        if (this.xfade >= 1) { this.src = this.nsrc; this.outGain = this.nsrc.outGain; this.modelId = this.nsrc.modelId; this.nsrc = null; }
      }
      const fs = this.fs;
      for (let k = 0; k < n; k++) {
        this.engGain += (this.engGainTarget - this.engGain) * this.gainK;
        this.starterLvl += (this.starterTarget - this.starterLvl) * this.gainK;
        this.speedS += (this.speed - this.speedS) * 0.0005; this.skidS += (this.skid - this.skidS) * 0.002;
        const g = this.engGain, w = wv[k];
        const eL = tL[k] * g, eR = tR[k] * g;
        const v = this.speedS, vr = v / 40;
        const road = Math.min(0.08, 0.05 * vr * vr);
        let wn = 0;   // wind: broadband, gusty, rising ~ v^3 in power
        if (v > 3) {
          this.windLfo += 2 * Math.PI * 0.3 / fs; if (this.windLfo > 6.2832) this.windLfo -= 6.2832;
          wn = bqs(bqs(this.rnd(), this.wind[0], this.windSt[0]), this.wind[1], this.windSt[1]);
          wn *= 0.02 * Math.pow(vr, 1.5) * (0.85 + 0.15 * Math.sin(this.windLfo));
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

  const api = { EngineSynth, EngineVoice, V1Source, ListenerSource, NpRandomState, Starter, TyreNoise, Limiter, bq, lopeOf, DEFAULT_SPEC };
  // ---- sound-model registry: v1 is built in, later versions are drop-in modules ----
  const ENGINES = { v1: { id: 'v1', Synth: EngineSynth, positions: [] } };
  (modules || []).forEach((mod) => { const m = mod(api); ENGINES[m.id] = m; });
  api.ENGINES = ENGINES;

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
}]);

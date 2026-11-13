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
        if (m.listen !== undefined) {   // view: 0 outside rear, 1 inside
          this.view = m.listen | 0; this.listen = this.view === 1 ? 1 : 0;
          if (this.src) this.src.setView(this.view); if (this.nsrc) this.nsrc.setView(this.view);
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
function engineModuleBrm8(core) {
    var bq = core.bq;
    var TWO_PI = 2 * Math.PI;
    var D = {"orders":[0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0,10.5,11.0,11.5,12.0,12.5,13.0,13.5,14.0,14.5,15.0,15.5,16.0,16.5,17.0,17.5,18.0,18.5,19.0,19.5,20.0,20.5,21.0,21.5,22.0,22.5,23.0,23.5,24.0,24.5,25.0,25.5,26.0,26.5,27.0,27.5,28.0,28.5,29.0,29.5,30.0,30.5,31.0,31.5,32.0,32.5,33.0,33.5,34.0,34.5,35.0,35.5,36.0,36.5,37.0,37.5,38.0,38.5,39.0,39.5,40.0,40.5,41.0,41.5,42.0,42.5,43.0,43.5,44.0,44.5,45.0,45.5,46.0,46.5,47.0,47.5,48.0],"centres":[3000,3250,3500,3750,4000,4250,4500,4750,5000,5250,5500,5750,6000,6250,6500,6750,7000,7250,7500,7750,8000,8250,8500,8750,9000,9250,9500,9750,10000,10250,10500],"whine":8.3,"rows":{"onboard":[[-31.494,-23.929,-10.868,-12.169,-4.583,-12.829,-15.35,-13.385,-18.816,-20.71,-15.072,-16.123,-17.755,-20.671,-20.402,-17.428,-19.559,-18.297,-15.017,-17.471,-19.922,-21.646,-22.506,-22.304,-22.554,-23.73,-23.499,-23.091,-24.422,-25.939,-25.8,-25.17,-22.455,-24.406,-25.008,-24.14,-25.012,-25.504,-25.315,-24.647,-25.126,-25.904,-26.361,-26.096,-26.119,-26.398,-25.961,-27.079,-26.115,-25.971,-26.503,-26.392,-27.15,-25.451,-26.778,-26.679,-27.307,-27.908,-28.097,-27.779,-27.541,-28.153,-28.34,-28.547,-28.519,-28.085,-28.382,-28.137,-29.487,-29.597,-29.025,-29.624,-29.415,-29.567,-29.107,-29.199,-28.996,-30.026,-29.668,-29.757,-29.761,-29.218,-28.982,-29.652,-30.149,-30.872,-31.112,-31.429,-31.401,-31.482,-30.916,-31.131,-31.438,-31.386,-31.58,-32.019],[-31.272,-23.546,-10.576,-12.128,-4.482,-13.381,-16.009,-13.383,-18.813,-20.465,-15.159,-16.449,-17.53,-20.413,-20.203,-17.25,-19.705,-18.525,-15.612,-17.99,-20.235,-21.987,-22.483,-22.35,-22.569,-23.904,-23.583,-23.164,-24.393,-25.929,-25.649,-25.25,-22.41,-24.194,-24.918,-23.735,-24.769,-25.264,-25.072,-24.457,-24.9,-25.693,-26.123,-25.873,-25.85,-26.166,-25.796,-26.814,-25.915,-25.892,-26.399,-26.289,-27.051,-25.413,-26.698,-26.601,-27.211,-27.727,-27.991,-27.692,-27.493,-27.986,-28.203,-28.458,-28.411,-27.992,-28.306,-28.077,-29.365,-29.543,-28.841,-29.545,-29.355,-29.527,-29.087,-29.095,-28.946,-30.023,-29.631,-29.765,-29.828,-29.246,-28.885,-29.707,-30.184,-30.958,-31.094,-31.367,-31.398,-31.491,-30.977,-31.214,-31.469,-31.44,-31.531,-32.012],[-32.017,-24.829,-11.698,-12.039,-4.949,-11.365,-14.123,-13.501,-18.655,-21.403,-14.789,-15.35,-18.681,-21.68,-21.059,-17.715,-18.94,-17.705,-13.669,-16.56,-19.527,-21.082,-22.569,-22.013,-22.507,-23.332,-23.172,-22.873,-24.309,-25.848,-26.229,-24.936,-22.685,-24.991,-25.244,-25.684,-25.742,-26.156,-26.139,-25.269,-25.744,-26.507,-27.111,-26.691,-26.957,-27.065,-26.417,-27.847,-26.739,-26.187,-26.831,-26.662,-27.417,-25.496,-26.989,-26.943,-27.588,-28.424,-28.383,-28.043,-27.738,-28.645,-28.687,-28.763,-28.784,-28.344,-28.586,-28.278,-29.795,-29.763,-29.612,-29.847,-29.558,-29.651,-29.209,-29.473,-29.165,-29.988,-29.744,-29.68,-29.598,-29.071,-29.308,-29.437,-30.066,-30.663,-31.195,-31.525,-31.424,-31.569,-30.861,-30.901,-31.399,-31.226,-31.683,-31.989],[-31.655,-24.46,-11.227,-11.791,-4.596,-12.025,-15.268,-13.536,-18.427,-21.179,-14.973,-16.026,-18.373,-21.201,-20.565,-17.783,-19.175,-18.312,-14.641,-17.601,-20.29,-21.722,-22.528,-22.143,-22.453,-23.644,-23.341,-23.088,-24.412,-25.958,-26.23,-25.292,-22.745,-24.773,-25.354,-25.142,-25.529,-25.939,-25.823,-24.886,-25.519,-26.246,-26.779,-26.315,-26.583,-26.735,-26.289,-27.486,-26.511,-26.169,-26.762,-26.525,-27.321,-25.529,-26.958,-26.849,-27.475,-28.183,-28.253,-27.92,-27.739,-28.353,-28.478,-28.656,-28.632,-28.308,-28.496,-28.173,-29.621,-29.747,-29.305,-29.794,-29.53,-29.629,-29.209,-29.265,-29.021,-30.016,-29.698,-29.704,-29.783,-29.073,-29.192,-29.561,-30.192,-30.9,-31.248,-31.471,-31.516,-31.709,-31.083,-31.086,-31.567,-31.43,-31.64,-31.992],[-31.841,-24.46,-11.708,-11.46,-5.174,-10.644,-14.457,-13.767,-18.231,-21.42,-14.595,-15.372,-19.518,-22.343,-21.269,-17.489,-18.265,-17.557,-13.432,-17.118,-20.371,-21.541,-22.498,-21.613,-22.538,-23.301,-22.872,-22.744,-23.905,-25.652,-26.436,-24.962,-22.989,-25.002,-25.233,-26.409,-25.85,-26.231,-26.51,-25.588,-25.777,-26.6,-27.338,-26.629,-27.104,-27.085,-26.445,-27.894,-26.88,-26.196,-26.957,-26.639,-27.441,-25.39,-26.996,-27.136,-27.643,-28.471,-28.384,-28.121,-27.92,-28.738,-28.604,-28.741,-28.699,-28.361,-28.598,-28.202,-29.757,-29.854,-29.815,-29.895,-29.541,-29.593,-29.371,-29.468,-29.31,-29.863,-29.705,-29.507,-29.601,-28.872,-29.462,-29.237,-30.076,-30.67,-31.277,-31.332,-31.441,-31.845,-31.08,-30.811,-31.502,-31.163,-31.635,-31.881],[-31.548,-23.856,-11.486,-10.842,-5.458,-9.891,-15.366,-14.061,-17.81,-21.272,-14.324,-15.749,-20.664,-22.934,-21.482,-17.202,-17.339,-17.425,-13.393,-18.421,-21.835,-22.22,-22.255,-21.129,-22.752,-23.216,-22.491,-22.418,-23.375,-25.534,-26.744,-25.109,-23.286,-24.8,-25.022,-26.695,-25.65,-26.126,-26.765,-26.09,-25.646,-26.599,-27.509,-26.478,-27.032,-26.851,-26.318,-27.707,-26.818,-26.097,-26.969,-26.554,-27.451,-25.262,-26.954,-27.428,-27.662,-28.386,-28.324,-28.173,-28.191,-28.818,-28.414,-28.725,-28.456,-28.218,-28.54,-27.995,-29.665,-30.044,-29.984,-29.933,-29.518,-29.472,-29.68,-29.473,-29.678,-29.681,-29.729,-29.242,-29.647,-28.655,-29.52,-28.981,-30.048,-30.699,-31.356,-31.024,-31.4,-32.192,-31.335,-30.665,-31.648,-31.087,-31.566,-31.782],[-30.628,-23.116,-10.704,-9.859,-4.88,-11.01,-17.969,-14.552,-16.85,-20.775,-15.118,-17.253,-20.004,-21.68,-20.138,-16.933,-18.153,-19.613,-16.201,-20.759,-22.508,-22.404,-22.012,-21.52,-22.107,-23.059,-22.755,-23.156,-23.916,-25.676,-26.63,-25.988,-23.63,-24.5,-25.452,-26.521,-25.463,-25.852,-26.408,-25.115,-25.441,-26.2,-26.692,-25.446,-26.223,-26.351,-26.037,-27.232,-26.716,-26.188,-26.734,-26.028,-27.029,-25.503,-27.041,-27.062,-27.119,-27.991,-27.793,-27.692,-27.559,-27.974,-27.798,-28.309,-28.121,-28.065,-28.296,-27.648,-29.199,-29.658,-29.381,-29.76,-29.481,-29.217,-29.256,-28.826,-28.967,-29.429,-29.2,-28.867,-29.484,-28.251,-29.191,-29.018,-30.214,-30.748,-31.134,-30.746,-31.31,-31.975,-31.25,-30.655,-31.721,-31.406,-31.454,-31.559],[-29.686,-22.116,-10.167,-8.665,-5.142,-11.637,-20.033,-15.41,-15.806,-19.95,-15.827,-18.168,-20.013,-20.852,-19.079,-16.17,-18.888,-21.957,-20.247,-22.18,-22.277,-21.813,-21.536,-21.67,-21.643,-22.345,-22.579,-23.908,-24.198,-25.322,-26.226,-26.511,-24.095,-24.275,-25.601,-26.297,-25.025,-25.555,-26.307,-24.826,-25.292,-25.867,-25.855,-24.352,-25.396,-25.848,-25.602,-26.939,-26.77,-26.183,-26.353,-25.3,-26.332,-25.416,-26.853,-26.759,-26.377,-27.57,-27.051,-27.162,-26.691,-27.218,-26.948,-27.673,-27.608,-27.735,-27.987,-27.217,-28.693,-28.995,-29.039,-29.413,-29.272,-28.709,-28.743,-28.19,-28.524,-28.853,-28.395,-28.129,-28.815,-27.562,-28.876,-28.688,-30.234,-30.362,-30.635,-30.119,-30.809,-31.498,-30.794,-30.207,-31.41,-31.245,-31.214,-31.078],[-30.516,-23.463,-10.476,-10.11,-3.683,-14.516,-21.084,-14.534,-16.706,-21.038,-16.45,-18.639,-18.39,-20.765,-19.246,-17.628,-21.243,-23.918,-22.121,-21.98,-22.526,-23.174,-22.56,-22.559,-21.334,-24.035,-23.982,-24.685,-25.134,-26.002,-26.401,-26.969,-23.823,-24.585,-26.452,-26.689,-26.32,-25.99,-26.002,-23.609,-25.344,-25.776,-26.144,-25.168,-26.118,-26.563,-26.35,-27.128,-26.702,-26.485,-26.974,-25.958,-26.973,-25.978,-27.317,-26.538,-26.923,-27.905,-27.862,-27.405,-27.089,-27.388,-27.955,-28.263,-28.583,-28.611,-28.374,-27.935,-28.999,-29.336,-28.68,-29.695,-29.507,-29.476,-28.721,-28.37,-27.859,-29.817,-28.949,-29.479,-29.877,-28.506,-29.187,-29.85,-30.71,-31.25,-31.173,-31.298,-31.822,-31.834,-31.371,-31.385,-31.961,-32.296,-31.55,-31.682],[-31.648,-25.623,-10.612,-12.564,-2.887,-18.495,-21.128,-13.394,-18.527,-22.289,-16.25,-18.635,-17.346,-20.476,-19.769,-19.598,-22.259,-23.674,-21.208,-21.81,-22.72,-24.354,-23.411,-23.499,-21.704,-25.919,-25.152,-24.604,-26.131,-27.209,-26.757,-26.821,-22.93,-24.889,-27.209,-25.978,-26.645,-26.176,-25.372,-23.266,-25.605,-26.079,-26.729,-26.433,-26.762,-26.858,-26.941,-27.242,-26.557,-26.757,-27.274,-26.982,-27.868,-26.408,-27.737,-26.437,-27.979,-28.298,-28.763,-27.963,-28.277,-27.99,-29.163,-29.037,-29.204,-29.127,-28.626,-28.461,-29.62,-30.173,-28.631,-30.135,-29.925,-30.384,-29.143,-29.019,-28.016,-31.054,-30.25,-30.95,-31.254,-29.764,-29.192,-30.864,-30.899,-32.373,-32.035,-32.721,-32.842,-32.462,-32.257,-32.677,-32.629,-33.122,-31.896,-32.512],[-31.964,-26.408,-10.434,-14.484,-2.95,-22.564,-20.421,-12.551,-20.017,-21.8,-15.636,-18.536,-16.446,-19.306,-19.394,-20.51,-22.085,-21.004,-19.865,-21.088,-21.915,-23.914,-23.049,-24.097,-22.834,-26.204,-25.197,-23.867,-26.375,-27.69,-26.357,-26.315,-22.066,-24.933,-26.783,-23.396,-25.24,-25.59,-24.562,-23.702,-25.415,-25.875,-26.163,-26.464,-26.249,-26.231,-26.685,-27.023,-26.105,-26.565,-26.688,-26.994,-27.76,-26.086,-27.195,-26.374,-27.911,-27.935,-28.62,-27.985,-28.566,-28.127,-28.921,-28.96,-28.776,-28.748,-28.461,-28.44,-29.844,-30.287,-28.525,-30.016,-29.882,-30.351,-29.436,-29.283,-28.786,-31.206,-30.674,-31.139,-31.346,-30.349,-28.817,-31.103,-30.772,-32.593,-32.017,-32.719,-32.493,-32.331,-32.227,-32.688,-32.476,-32.742,-31.941,-32.82],[-31.573,-25.293,-10.089,-15.1,-3.602,-25.561,-19.372,-12.121,-20.84,-20.115,-14.895,-18.67,-15.592,-18.265,-18.82,-19.931,-21.359,-18.084,-18.792,-19.674,-20.62,-22.966,-22.109,-24.109,-24.077,-25.34,-24.448,-23.034,-25.545,-26.91,-25.437,-25.971,-21.536,-24.182,-24.897,-20.821,-23.588,-24.66,-23.674,-24.172,-24.498,-25.076,-25.124,-25.601,-25.004,-25.32,-25.62,-26.232,-24.979,-25.666,-25.808,-26.162,-26.812,-25.285,-26.003,-26.322,-26.856,-26.987,-27.82,-27.451,-27.703,-27.584,-27.711,-28.308,-27.952,-27.776,-28.097,-28.067,-29.471,-29.614,-27.988,-29.353,-29.278,-29.56,-29.284,-28.987,-29.556,-30.494,-29.989,-30.483,-30.522,-30.364,-28.299,-30.642,-30.371,-31.861,-31.125,-31.536,-31.343,-31.492,-31.321,-31.932,-31.638,-31.689,-31.603,-32.566],[-30.686,-22.535,-9.425,-13.746,-4.849,-26.002,-18.315,-12.343,-21.122,-18.145,-14.561,-18.263,-15.116,-18.045,-18.88,-17.954,-20.27,-16.353,-18.193,-18.344,-19.525,-22.504,-21.444,-23.0,-24.155,-24.285,-23.567,-22.681,-23.808,-25.359,-24.63,-25.607,-21.217,-22.422,-22.652,-19.364,-22.336,-23.789,-22.82,-23.872,-23.232,-24.12,-24.256,-24.608,-23.533,-24.393,-23.99,-24.931,-23.465,-24.384,-25.051,-25.184,-25.684,-24.402,-25.046,-26.131,-25.882,-25.968,-26.844,-26.729,-26.435,-26.687,-26.632,-27.592,-27.208,-26.627,-27.674,-27.512,-28.56,-28.604,-27.217,-28.418,-28.364,-28.652,-28.726,-28.45,-29.569,-29.671,-29.008,-29.608,-29.585,-29.915,-27.71,-29.649,-29.814,-30.689,-30.083,-30.204,-30.27,-30.457,-30.109,-31.135,-30.616,-30.602,-30.898,-31.899],[-29.31,-19.23,-8.141,-11.687,-7.048,-24.135,-17.666,-13.573,-20.984,-16.267,-15.203,-16.617,-15.307,-18.619,-19.777,-14.797,-19.3,-16.461,-18.237,-17.905,-19.021,-22.659,-21.42,-20.889,-23.344,-23.533,-23.119,-22.806,-21.94,-24.12,-23.665,-24.253,-20.924,-20.726,-21.174,-19.328,-21.411,-22.964,-22.176,-23.165,-22.086,-23.271,-23.421,-23.542,-22.263,-23.435,-22.482,-23.631,-22.546,-23.446,-24.448,-24.454,-25.069,-23.794,-24.804,-25.737,-25.491,-25.263,-25.915,-26.19,-25.385,-25.994,-26.152,-26.943,-26.685,-25.685,-27.19,-27.031,-27.429,-27.671,-26.8,-27.583,-27.587,-28.088,-28.272,-28.076,-28.994,-28.925,-28.332,-28.72,-28.775,-29.133,-27.317,-28.696,-29.368,-29.633,-29.42,-29.27,-29.67,-29.641,-29.175,-30.534,-29.901,-29.877,-30.179,-31.173],[-27.952,-17.072,-6.672,-10.706,-10.455,-22.214,-17.976,-15.861,-20.775,-15.49,-16.708,-15.122,-15.88,-19.445,-21.275,-11.49,-19.147,-18.539,-19.203,-18.617,-19.286,-23.124,-22.396,-19.588,-22.931,-23.163,-23.071,-22.977,-21.066,-23.741,-21.832,-22.38,-20.977,-20.168,-21.124,-20.437,-20.728,-21.91,-22.12,-23.008,-21.187,-22.721,-22.797,-22.627,-21.872,-22.9,-22.024,-22.932,-23.026,-23.464,-24.098,-24.253,-25.144,-23.628,-25.105,-25.278,-25.556,-25.257,-25.653,-26.204,-25.104,-26.055,-26.345,-26.565,-26.643,-25.231,-26.736,-26.922,-26.799,-27.416,-27.173,-27.474,-27.548,-28.174,-28.233,-28.065,-28.519,-28.449,-28.194,-28.281,-28.318,-28.688,-27.434,-28.48,-29.332,-29.239,-29.361,-29.133,-29.747,-29.499,-29.163,-30.363,-29.796,-29.838,-30.025,-30.873],[-27.212,-16.266,-5.976,-11.111,-13.172,-20.851,-19.059,-17.503,-20.729,-16.2,-17.62,-14.842,-15.974,-19.699,-22.237,-9.438,-19.825,-20.796,-20.425,-19.351,-19.609,-23.678,-23.677,-20.081,-22.669,-23.283,-23.04,-22.842,-21.043,-23.325,-20.198,-21.528,-21.265,-20.256,-21.752,-21.174,-20.322,-20.771,-22.616,-23.183,-20.433,-22.33,-22.778,-22.356,-22.389,-22.962,-22.464,-22.831,-24.142,-24.089,-24.042,-24.314,-25.094,-23.714,-25.3,-24.962,-25.443,-25.535,-26.043,-26.543,-25.43,-26.533,-26.677,-26.56,-26.983,-25.153,-26.514,-27.107,-27.048,-27.911,-27.818,-27.88,-27.927,-28.397,-28.117,-28.227,-28.351,-28.466,-28.332,-28.432,-28.321,-28.753,-27.671,-28.877,-29.427,-29.45,-29.632,-29.696,-30.117,-29.863,-29.771,-30.341,-29.983,-30.194,-30.369,-30.909],[-26.717,-15.717,-6.009,-11.934,-13.802,-19.38,-19.806,-17.406,-20.123,-17.136,-17.64,-15.245,-16.057,-19.556,-21.966,-8.734,-20.578,-21.778,-20.956,-19.187,-19.754,-23.949,-23.582,-21.36,-21.978,-23.797,-22.896,-22.009,-20.878,-22.345,-19.482,-21.297,-21.068,-20.257,-21.665,-21.129,-20.374,-19.451,-22.78,-22.877,-19.886,-21.656,-22.978,-22.497,-23.012,-23.073,-22.748,-22.97,-24.494,-24.364,-23.833,-24.165,-24.691,-23.754,-25.18,-24.787,-25.105,-25.343,-26.197,-26.531,-25.639,-26.443,-26.554,-26.547,-27.105,-25.175,-26.441,-27.265,-27.385,-28.159,-27.725,-27.826,-27.94,-28.089,-27.619,-28.081,-28.26,-28.522,-28.45,-28.441,-28.28,-28.666,-27.573,-29.111,-29.194,-29.577,-29.645,-30.133,-29.982,-29.939,-30.021,-29.981,-29.978,-30.349,-30.544,-30.765],[-26.107,-15.116,-6.492,-12.543,-13.528,-18.046,-19.687,-16.353,-19.065,-17.201,-17.406,-15.632,-16.48,-19.315,-21.105,-9.023,-20.725,-22.111,-21.096,-18.462,-20.223,-23.581,-22.502,-22.226,-21.334,-24.125,-22.387,-20.56,-20.554,-21.295,-19.431,-20.767,-20.366,-20.382,-20.871,-20.994,-20.912,-17.688,-22.192,-22.107,-19.609,-20.9,-22.876,-22.731,-23.098,-22.957,-22.615,-23.154,-24.093,-24.044,-23.219,-23.932,-24.198,-23.648,-24.844,-24.601,-24.729,-24.728,-25.92,-26.07,-25.505,-25.699,-25.991,-26.459,-26.902,-25.227,-26.333,-27.105,-27.052,-27.571,-27.009,-27.216,-27.652,-27.403,-26.968,-27.509,-28.102,-28.351,-28.458,-28.042,-27.876,-28.209,-27.148,-28.902,-28.719,-29.333,-29.204,-30.033,-29.37,-29.635,-29.797,-29.492,-29.716,-30.206,-30.391,-30.436],[-25.404,-14.901,-7.08,-13.056,-13.174,-17.338,-19.285,-14.921,-18.395,-16.723,-17.352,-15.635,-16.367,-19.011,-20.591,-9.969,-20.254,-22.262,-21.184,-17.912,-20.919,-22.94,-21.533,-22.257,-21.04,-23.757,-21.534,-19.183,-20.529,-20.676,-19.739,-19.969,-19.61,-20.755,-20.031,-20.876,-21.478,-16.272,-21.402,-21.452,-19.587,-20.57,-22.404,-22.868,-22.617,-22.797,-22.413,-23.309,-23.538,-23.539,-22.556,-23.887,-23.827,-23.56,-24.362,-24.394,-24.326,-24.153,-25.529,-25.51,-25.259,-24.995,-25.383,-26.439,-26.632,-25.191,-26.105,-26.561,-26.339,-26.646,-26.401,-26.615,-27.322,-26.778,-26.472,-26.932,-27.969,-28.206,-28.267,-27.659,-27.383,-27.642,-26.535,-28.565,-28.32,-28.958,-28.66,-29.739,-28.857,-29.448,-29.507,-29.277,-29.399,-29.964,-30.192,-30.212],[-24.729,-15.306,-6.667,-13.989,-13.353,-17.511,-19.596,-13.615,-18.546,-16.707,-17.723,-15.739,-14.856,-18.995,-20.739,-11.201,-19.895,-22.453,-21.207,-18.312,-21.456,-22.604,-21.142,-22.099,-20.596,-22.766,-21.068,-18.526,-21.196,-20.782,-20.177,-19.73,-19.282,-21.353,-19.797,-20.69,-21.406,-16.174,-21.205,-21.637,-19.988,-21.023,-22.168,-22.915,-22.472,-22.976,-22.598,-23.699,-23.375,-23.409,-22.501,-24.265,-23.916,-23.776,-23.994,-24.52,-24.176,-24.185,-25.48,-25.351,-25.448,-25.112,-25.207,-26.314,-26.475,-25.03,-25.915,-26.095,-26.013,-26.231,-26.378,-26.572,-27.112,-26.561,-26.529,-26.898,-28.091,-28.385,-28.073,-27.824,-27.331,-27.464,-26.233,-28.579,-28.316,-28.852,-28.514,-29.747,-28.888,-29.665,-29.517,-29.605,-29.393,-29.955,-30.349,-30.369],[-24.014,-15.261,-4.957,-15.641,-14.102,-18.567,-21.161,-13.088,-19.201,-17.856,-18.414,-16.951,-13.6,-19.788,-21.036,-12.758,-20.301,-22.684,-21.582,-20.004,-22.195,-22.863,-21.444,-22.563,-19.68,-21.607,-21.585,-19.096,-22.559,-21.357,-20.403,-20.572,-19.732,-22.265,-20.492,-20.828,-21.214,-17.797,-22.027,-22.922,-21.097,-22.318,-22.863,-23.463,-23.516,-23.734,-23.527,-24.542,-24.035,-24.007,-23.504,-25.112,-24.656,-24.486,-24.285,-25.381,-24.748,-25.086,-26.084,-26.084,-26.479,-26.386,-25.839,-25.956,-26.575,-25.242,-26.214,-26.292,-26.467,-26.718,-27.038,-27.255,-27.179,-27.023,-27.495,-27.645,-28.648,-28.906,-28.369,-28.717,-28.089,-28.178,-26.96,-29.209,-28.923,-29.286,-28.986,-30.266,-29.653,-30.125,-29.936,-30.467,-30.054,-30.524,-31.07,-31.002],[-23.199,-14.089,-3.447,-17.396,-14.163,-20.017,-22.69,-13.648,-19.782,-19.954,-19.359,-19.034,-14.157,-20.966,-21.338,-14.796,-21.603,-23.049,-22.614,-22.14,-23.26,-23.657,-22.316,-23.459,-19.379,-21.002,-22.603,-20.838,-23.767,-21.895,-20.502,-22.086,-21.014,-23.328,-21.834,-21.818,-21.88,-20.506,-23.337,-24.392,-22.571,-23.804,-24.352,-24.746,-25.234,-25.009,-25.108,-25.301,-25.291,-25.124,-25.152,-25.967,-25.74,-25.375,-25.354,-26.665,-25.976,-26.479,-27.014,-27.409,-27.78,-27.946,-27.076,-26.137,-27.187,-26.156,-27.198,-27.254,-27.64,-27.676,-28.162,-28.339,-27.77,-28.207,-29.08,-28.908,-29.508,-29.634,-29.302,-29.864,-29.48,-29.577,-28.726,-30.283,-30.007,-30.237,-29.974,-31.241,-30.942,-30.705,-30.746,-31.517,-31.319,-31.552,-32.132,-31.841],[-22.529,-13.188,-2.666,-18.796,-12.955,-21.186,-22.056,-14.744,-20.332,-21.989,-20.428,-19.963,-16.165,-21.755,-22.176,-16.893,-23.11,-23.666,-23.565,-23.432,-23.857,-24.764,-23.458,-24.056,-19.694,-21.317,-23.064,-22.647,-24.145,-22.389,-21.091,-23.671,-22.689,-24.2,-23.278,-23.447,-23.005,-22.909,-24.274,-25.38,-23.77,-24.952,-25.736,-25.931,-26.462,-26.354,-26.718,-25.626,-26.292,-26.18,-26.597,-26.632,-26.834,-26.03,-26.725,-27.681,-27.446,-27.815,-27.78,-28.155,-28.518,-28.818,-28.114,-27.303,-28.042,-27.255,-28.354,-28.462,-28.982,-28.598,-29.403,-29.447,-28.846,-29.506,-30.414,-30.133,-30.344,-30.311,-30.387,-30.724,-30.825,-30.785,-30.463,-31.233,-30.961,-31.275,-31.128,-32.196,-32.236,-31.558,-31.832,-32.491,-32.501,-32.48,-33.133,-32.549],[-22.148,-13.014,-2.455,-20.105,-11.368,-21.541,-20.19,-15.484,-20.547,-23.109,-20.434,-19.586,-18.174,-22.418,-23.232,-18.837,-23.694,-24.219,-23.828,-23.429,-23.829,-25.849,-23.904,-24.161,-19.856,-22.371,-22.554,-23.555,-23.812,-22.535,-21.895,-24.915,-23.61,-24.834,-24.37,-24.967,-23.674,-24.018,-25.005,-26.098,-24.578,-25.981,-26.172,-26.577,-27.066,-27.086,-27.631,-25.97,-26.827,-26.897,-27.327,-27.183,-27.694,-26.444,-27.967,-28.089,-28.816,-28.467,-28.272,-28.112,-28.693,-29.299,-28.518,-28.949,-28.556,-28.187,-29.122,-29.476,-29.758,-29.408,-30.267,-30.216,-29.728,-30.242,-31.012,-30.792,-30.957,-30.731,-31.14,-31.054,-31.586,-31.353,-31.318,-31.68,-31.259,-31.888,-31.909,-32.584,-33.092,-32.611,-32.695,-33.11,-33.044,-33.138,-33.721,-33.059],[-22.259,-13.535,-2.485,-21.241,-10.291,-21.494,-18.892,-15.829,-20.013,-22.719,-18.834,-19.584,-19.0,-23.133,-23.511,-20.613,-23.504,-24.574,-23.88,-23.106,-23.673,-26.637,-22.679,-24.208,-20.426,-23.496,-21.872,-23.744,-23.357,-22.491,-22.377,-25.282,-23.293,-25.175,-25.004,-25.739,-23.876,-24.095,-25.521,-26.322,-25.105,-26.864,-25.903,-26.952,-26.946,-27.122,-27.679,-26.532,-27.033,-27.21,-27.289,-27.323,-28.254,-26.639,-28.759,-28.184,-29.49,-28.401,-28.393,-27.866,-28.51,-29.682,-28.604,-30.167,-28.974,-29.134,-29.395,-30.262,-30.074,-30.078,-30.586,-30.611,-30.07,-30.644,-31.178,-30.881,-31.347,-31.121,-31.399,-31.042,-31.926,-31.422,-31.649,-31.852,-31.241,-32.129,-32.195,-32.567,-33.252,-33.53,-33.152,-33.387,-33.247,-33.548,-33.744,-33.35],[-23.23,-15.197,-2.39,-22.26,-9.964,-22.15,-18.862,-16.734,-19.008,-21.613,-17.071,-20.665,-19.234,-23.493,-22.725,-21.313,-23.079,-24.843,-24.636,-23.644,-23.757,-27.263,-21.163,-24.963,-22.094,-23.822,-21.843,-23.722,-23.109,-22.71,-22.551,-24.962,-22.612,-25.172,-25.099,-25.282,-24.442,-23.78,-24.888,-25.875,-25.475,-27.711,-25.975,-27.176,-26.162,-27.106,-27.351,-26.984,-27.002,-27.356,-27.279,-26.954,-28.88,-26.621,-29.236,-28.435,-28.948,-28.353,-28.371,-27.787,-28.282,-29.962,-29.072,-30.569,-29.936,-30.289,-29.346,-30.711,-30.615,-30.434,-30.865,-31.081,-30.513,-31.177,-31.45,-30.85,-31.518,-31.778,-31.443,-31.217,-32.205,-31.524,-32.272,-32.105,-31.591,-32.364,-32.159,-32.7,-32.846,-33.953,-33.737,-33.749,-33.543,-33.787,-33.512,-33.684],[-23.856,-16.172,-2.292,-22.347,-10.188,-22.857,-19.493,-17.412,-18.567,-21.541,-16.999,-21.368,-19.121,-23.341,-21.978,-20.873,-22.248,-24.981,-25.257,-24.169,-24.024,-27.016,-21.441,-25.53,-22.628,-23.305,-22.217,-23.572,-22.631,-23.001,-22.12,-24.782,-22.682,-24.986,-24.458,-24.383,-25.065,-23.255,-23.833,-25.255,-25.385,-27.889,-26.168,-26.984,-25.903,-27.052,-27.081,-26.9,-26.876,-27.481,-27.672,-26.751,-29.062,-26.653,-29.399,-28.506,-28.02,-28.297,-28.682,-27.948,-28.354,-29.946,-29.402,-30.121,-30.452,-30.524,-29.17,-30.428,-30.996,-29.976,-31.021,-31.356,-30.862,-31.251,-31.591,-30.905,-31.187,-32.098,-31.557,-31.535,-32.141,-31.765,-32.684,-32.194,-31.979,-32.365,-31.93,-32.92,-32.803,-33.682,-34.173,-34.001,-33.711,-34.123,-33.629,-34.11],[-23.742,-15.812,-2.322,-21.789,-10.425,-22.721,-19.75,-17.134,-18.671,-21.671,-17.36,-21.198,-18.733,-23.151,-21.877,-20.251,-22.084,-24.886,-25.077,-24.089,-24.025,-26.627,-21.815,-25.351,-22.193,-23.017,-22.324,-23.442,-22.553,-23.021,-21.873,-24.682,-22.773,-24.865,-24.188,-24.196,-24.911,-23.037,-23.695,-25.133,-25.174,-27.49,-26.113,-26.81,-25.938,-26.962,-26.976,-26.71,-26.792,-27.377,-27.633,-26.74,-28.78,-26.603,-29.088,-28.396,-27.795,-28.21,-28.682,-27.982,-28.381,-29.79,-29.245,-29.669,-30.178,-30.053,-29.053,-30.119,-30.794,-29.695,-30.828,-31.152,-30.676,-31.003,-31.455,-30.85,-30.969,-31.899,-31.442,-31.511,-31.957,-31.685,-32.423,-32.067,-31.91,-32.222,-31.806,-32.864,-32.794,-33.373,-33.957,-33.871,-33.591,-34.035,-33.64,-34.018],[-23.494,-15.151,-2.543,-20.211,-10.858,-21.907,-19.98,-16.307,-18.958,-21.304,-17.825,-20.35,-17.818,-22.577,-21.916,-18.24,-22.084,-24.483,-24.392,-23.361,-23.766,-25.94,-22.065,-24.794,-21.453,-22.716,-22.322,-22.861,-22.692,-22.752,-21.631,-24.098,-22.498,-24.478,-23.681,-23.823,-24.18,-22.285,-23.681,-24.959,-24.439,-26.321,-25.696,-26.318,-25.794,-26.52,-26.558,-26.298,-26.456,-26.842,-26.973,-26.586,-27.962,-26.308,-28.147,-27.942,-27.496,-27.813,-28.288,-27.811,-28.196,-29.235,-28.674,-28.943,-29.398,-28.973,-28.694,-29.477,-29.991,-29.256,-30.175,-30.423,-30.017,-30.314,-30.811,-30.408,-30.649,-31.294,-30.981,-31.071,-31.342,-31.16,-31.365,-31.631,-31.422,-31.781,-31.448,-32.489,-32.408,-32.754,-33.135,-33.248,-33.019,-33.405,-33.312,-33.447],[-23.708,-15.6,-2.428,-20.892,-10.681,-22.353,-19.957,-16.75,-18.746,-21.463,-17.588,-20.831,-18.181,-22.838,-21.775,-19.064,-21.931,-24.7,-24.765,-23.746,-23.913,-26.195,-21.986,-25.099,-21.863,-22.801,-22.36,-23.098,-22.513,-22.937,-21.658,-24.339,-22.629,-24.628,-23.824,-23.898,-24.586,-22.521,-23.55,-24.94,-24.766,-26.855,-25.891,-26.515,-25.81,-26.707,-26.692,-26.479,-26.591,-27.105,-27.31,-26.631,-28.341,-26.447,-28.575,-28.153,-27.5,-27.97,-28.515,-27.905,-28.281,-29.483,-28.955,-29.191,-29.813,-29.475,-28.827,-29.726,-30.392,-29.388,-30.469,-30.777,-30.35,-30.619,-31.112,-30.617,-30.725,-31.608,-31.19,-31.319,-31.607,-31.417,-31.867,-31.829,-31.692,-31.974,-31.575,-32.68,-32.58,-32.998,-33.558,-33.566,-33.299,-33.747,-33.479,-33.766],[-23.604,-14.939,-2.989,-18.021,-11.341,-20.93,-20.091,-15.734,-19.085,-20.299,-18.002,-19.174,-17.023,-21.772,-21.776,-15.586,-21.781,-23.903,-23.56,-22.166,-23.164,-25.183,-22.132,-24.115,-21.094,-22.602,-22.263,-22.064,-22.518,-22.422,-21.263,-23.188,-21.939,-23.686,-22.904,-23.131,-23.373,-20.978,-23.366,-24.395,-23.313,-24.892,-24.994,-25.509,-25.214,-25.721,-25.667,-25.668,-25.868,-26.102,-25.994,-26.134,-26.981,-25.776,-27.129,-27.202,-26.855,-27.094,-27.718,-27.435,-27.69,-28.371,-27.954,-28.208,-28.659,-27.895,-28.126,-28.728,-29.08,-28.696,-29.329,-29.545,-29.321,-29.49,-29.86,-29.699,-30.114,-30.586,-30.342,-30.389,-30.451,-30.399,-30.104,-30.981,-30.771,-31.165,-30.879,-31.911,-31.698,-31.984,-32.215,-32.383,-32.205,-32.592,-32.688,-32.734]],"paddock":[[-18.96,-13.014,-7.439,-5.487,-6.865,-13.006,-15.444,-10.442,-16.737,-20.588,-19.319,-17.933,-20.707,-24.358,-27.586,-27.254,-28.458,-26.253,-25.959,-24.972,-24.923,-29.708,-30.296,-30.283,-30.184,-30.794,-30.448,-32.481,-33.701,-33.406,-33.531,-35.468,-33.962,-34.029,-34.788,-34.548,-36.647,-36.553,-35.712,-36.864,-37.414,-38.728,-38.721,-38.625,-39.723,-39.274,-38.578,-39.139,-40.532,-40.656,-39.764,-39.985,-40.771,-39.566,-40.787,-40.833,-41.003,-41.541,-42.354,-42.885,-42.828,-43.527,-42.534,-43.722,-44.145,-44.885,-44.657,-44.527,-44.789,-44.808,-45.391,-45.067,-46.471,-44.931,-44.901,-45.184,-45.038,-45.545,-45.617,-45.267,-44.527,-44.853,-44.811,-45.165,-46.109,-46.02,-46.177,-46.094,-46.183,-46.888,-47.037,-46.679,-47.697,-47.399,-48.658,-48.499],[-19.305,-13.475,-7.708,-5.542,-7.014,-12.839,-15.292,-9.754,-16.558,-20.425,-19.124,-17.112,-20.154,-24.034,-27.567,-27.161,-28.454,-26.068,-25.437,-24.391,-24.07,-29.485,-30.465,-30.337,-30.221,-30.609,-29.905,-32.158,-33.512,-33.188,-33.505,-35.399,-33.68,-33.69,-34.375,-34.155,-36.385,-36.25,-35.302,-36.433,-36.993,-38.558,-38.623,-38.469,-39.493,-39.069,-38.348,-38.955,-40.484,-40.624,-39.477,-39.677,-40.482,-39.001,-40.449,-40.351,-40.506,-40.942,-41.872,-42.427,-42.376,-43.186,-41.977,-43.324,-43.714,-44.58,-44.424,-44.203,-44.516,-44.461,-45.165,-44.85,-46.452,-44.539,-44.571,-44.813,-44.892,-45.527,-45.514,-45.16,-44.354,-44.714,-44.655,-44.933,-46.079,-45.95,-45.91,-45.851,-45.838,-46.594,-46.648,-46.316,-47.381,-46.997,-48.412,-48.296],[-19.465,-13.728,-7.828,-5.489,-7.023,-12.706,-15.234,-9.67,-16.588,-20.322,-19.067,-17.175,-20.028,-24.066,-27.533,-27.099,-28.455,-25.927,-25.192,-24.284,-24.343,-29.635,-30.549,-30.38,-30.181,-30.439,-29.72,-32.07,-33.493,-33.171,-33.528,-35.369,-33.651,-33.595,-34.313,-34.048,-36.193,-36.177,-35.201,-36.381,-37.024,-38.589,-38.638,-38.581,-39.457,-39.049,-38.459,-38.926,-40.478,-40.67,-39.452,-39.556,-40.35,-38.783,-40.277,-40.249,-40.257,-40.868,-41.792,-42.371,-42.421,-43.084,-41.999,-43.363,-43.65,-44.531,-44.313,-44.147,-44.469,-44.489,-45.102,-44.765,-46.434,-44.464,-44.499,-44.722,-44.844,-45.546,-45.561,-45.176,-44.367,-44.743,-44.668,-44.898,-46.138,-45.909,-45.826,-45.768,-45.772,-46.41,-46.431,-46.104,-47.244,-46.761,-48.201,-48.108],[-18.695,-12.753,-7.288,-5.215,-6.673,-12.96,-15.689,-11.718,-17.322,-20.613,-19.548,-19.708,-21.446,-24.974,-27.556,-27.269,-28.427,-26.242,-26.469,-26.015,-27.451,-30.499,-30.342,-30.312,-29.965,-30.707,-31.146,-32.853,-34.114,-33.856,-33.54,-35.558,-34.319,-34.336,-35.331,-35.097,-36.801,-37.115,-36.508,-37.668,-38.368,-39.307,-39.065,-39.312,-40.182,-39.7,-39.298,-39.435,-40.743,-40.791,-40.134,-40.309,-40.911,-40.16,-41.008,-41.549,-41.457,-42.463,-43.044,-43.609,-43.692,-43.876,-43.585,-44.451,-44.863,-45.437,-44.96,-45.037,-45.201,-45.421,-45.673,-45.275,-46.506,-45.466,-45.41,-45.683,-45.111,-45.669,-45.924,-45.449,-44.789,-45.034,-45.052,-45.377,-46.088,-45.857,-46.253,-46.103,-46.387,-46.876,-47.147,-46.758,-47.825,-47.544,-48.608,-48.347],[-18.175,-12.045,-6.842,-5.263,-6.53,-13.403,-15.862,-13.145,-17.44,-20.98,-19.831,-21.467,-22.531,-25.403,-27.635,-27.498,-28.477,-26.656,-27.604,-27.154,-29.125,-30.644,-30.039,-30.202,-30.067,-31.239,-32.263,-33.453,-34.363,-34.101,-33.578,-35.652,-34.765,-34.983,-36.087,-35.84,-37.456,-37.639,-37.212,-38.405,-39.043,-39.509,-39.186,-39.459,-40.585,-39.999,-39.404,-39.716,-40.763,-40.811,-40.608,-40.89,-41.563,-41.401,-41.752,-42.48,-42.579,-43.676,-43.95,-44.455,-44.413,-44.536,-44.523,-45.014,-45.549,-45.853,-45.37,-45.551,-45.617,-45.92,-46.07,-45.668,-46.531,-46.12,-45.901,-46.325,-45.4,-45.639,-45.941,-45.545,-44.962,-45.155,-45.169,-45.705,-46.102,-46.07,-46.821,-46.647,-47.097,-47.595,-48.066,-47.604,-48.492,-48.434,-49.224,-48.914],[-17.919,-11.625,-6.52,-5.565,-6.444,-13.878,-15.818,-13.338,-16.922,-21.338,-19.988,-22.069,-23.128,-25.429,-27.782,-27.683,-28.565,-27.16,-28.299,-27.206,-28.911,-30.169,-29.545,-30.082,-30.364,-31.838,-32.719,-33.765,-34.115,-33.921,-33.706,-35.685,-35.125,-35.569,-36.56,-36.043,-37.759,-37.53,-37.013,-38.423,-38.862,-39.031,-38.89,-38.851,-40.418,-39.835,-39.047,-39.726,-40.503,-40.774,-40.957,-41.23,-42.193,-42.225,-42.324,-42.811,-43.261,-44.303,-44.411,-44.735,-44.599,-44.964,-44.825,-45.219,-45.689,-45.807,-45.434,-45.646,-45.75,-46.135,-46.235,-45.909,-46.521,-46.436,-45.981,-46.576,-45.688,-45.45,-45.752,-45.586,-45.119,-45.408,-45.334,-46.147,-46.413,-46.635,-47.602,-47.486,-48.03,-48.484,-49.071,-48.562,-49.28,-49.387,-49.895,-49.681],[-18.299,-11.811,-6.557,-6.124,-6.223,-14.134,-15.622,-11.86,-15.847,-21.558,-20.093,-21.31,-23.154,-25.138,-27.497,-27.476,-28.452,-27.638,-27.86,-25.539,-26.715,-28.693,-28.822,-29.663,-30.23,-31.796,-32.039,-33.418,-33.284,-33.462,-33.867,-35.477,-35.069,-35.751,-36.237,-35.067,-36.692,-36.042,-35.197,-36.928,-36.852,-37.104,-37.48,-36.743,-38.901,-38.737,-38.194,-39.195,-39.793,-40.212,-40.704,-40.717,-42.114,-41.901,-42.047,-41.75,-42.486,-43.57,-43.772,-43.706,-43.792,-44.804,-44.219,-44.982,-45.184,-45.192,-44.657,-45.116,-45.277,-45.73,-45.66,-45.656,-46.285,-46.421,-45.708,-46.212,-45.869,-45.258,-45.691,-45.798,-45.569,-46.06,-45.833,-46.927,-47.12,-47.597,-48.584,-48.597,-49.095,-49.415,-49.975,-49.513,-50.106,-50.442,-50.499,-50.482],[-20.622,-13.626,-7.576,-6.595,-5.786,-13.409,-14.988,-9.932,-14.601,-21.311,-19.998,-19.932,-22.779,-24.616,-25.504,-26.56,-27.556,-27.206,-25.456,-22.757,-23.836,-26.39,-27.958,-28.316,-28.541,-30.192,-30.406,-32.11,-32.297,-32.902,-33.806,-34.508,-33.614,-34.684,-34.484,-32.881,-33.971,-32.773,-31.947,-33.64,-33.491,-34.112,-34.936,-33.618,-36.124,-36.634,-36.753,-37.774,-38.423,-38.347,-38.978,-38.513,-40.373,-39.797,-40.13,-39.043,-39.821,-41.272,-41.625,-41.283,-41.747,-43.617,-42.517,-43.917,-43.842,-43.579,-42.401,-43.479,-43.522,-43.996,-43.917,-44.397,-45.425,-46.231,-45.198,-45.132,-45.981,-45.69,-46.415,-46.598,-46.528,-47.12,-46.722,-48.062,-47.985,-48.81,-49.6,-49.732,-49.809,-50.178,-50.479,-50.283,-50.602,-51.622,-50.88,-51.06],[-25.77,-16.455,-8.799,-6.161,-5.721,-12.282,-13.73,-9.965,-14.27,-20.774,-19.468,-19.563,-22.693,-24.315,-23.715,-25.482,-25.9,-25.879,-23.538,-21.662,-23.194,-25.26,-27.267,-27.145,-27.087,-29.135,-29.564,-31.143,-32.305,-32.793,-33.375,-33.591,-32.323,-33.651,-33.277,-31.764,-32.212,-30.61,-30.409,-31.87,-32.038,-32.868,-33.729,-32.191,-34.774,-35.437,-36.116,-36.681,-37.519,-36.733,-37.457,-36.926,-38.85,-38.32,-38.414,-37.686,-38.272,-39.95,-40.465,-40.124,-40.605,-42.329,-41.374,-42.998,-43.03,-42.146,-40.91,-42.214,-42.253,-42.955,-43.222,-43.713,-44.991,-46.553,-45.343,-44.776,-46.533,-46.933,-47.69,-47.704,-47.368,-47.879,-47.349,-48.793,-48.505,-49.437,-50.16,-50.171,-49.972,-50.624,-50.481,-50.56,-50.659,-52.092,-50.909,-51.163],[-28.726,-15.981,-8.233,-5.834,-6.113,-12.331,-12.426,-11.337,-14.968,-20.567,-18.669,-19.713,-22.719,-24.605,-23.475,-24.435,-24.009,-24.728,-23.176,-22.514,-24.41,-25.321,-26.556,-26.788,-27.266,-29.635,-29.592,-30.903,-32.835,-33.051,-32.849,-33.37,-32.27,-33.574,-33.219,-31.825,-31.831,-30.315,-30.928,-32.18,-32.44,-33.467,-33.976,-32.646,-35.051,-35.342,-36.596,-36.433,-37.505,-36.118,-36.931,-36.869,-38.567,-38.313,-37.915,-38.219,-38.294,-40.0,-40.836,-40.511,-40.918,-41.547,-41.001,-42.957,-43.166,-41.982,-41.308,-42.422,-42.469,-43.597,-44.213,-44.424,-45.697,-47.332,-46.353,-45.552,-47.461,-47.718,-48.164,-48.332,-47.903,-48.491,-47.717,-49.101,-48.955,-49.596,-50.505,-50.281,-50.402,-51.022,-50.538,-50.639,-50.818,-51.812,-51.036,-51.408],[-28.599,-14.463,-6.937,-6.124,-6.51,-13.093,-11.546,-12.719,-16.438,-20.632,-17.992,-19.688,-22.569,-25.125,-24.008,-23.425,-22.079,-23.394,-23.316,-24.26,-25.72,-25.684,-25.849,-26.825,-28.26,-30.719,-30.146,-30.731,-32.914,-33.097,-32.293,-33.064,-32.174,-33.274,-33.062,-31.707,-31.943,-31.012,-32.242,-33.278,-33.379,-34.683,-34.564,-33.971,-35.738,-35.538,-37.228,-36.591,-37.818,-36.086,-36.847,-37.474,-38.888,-38.91,-38.321,-39.488,-38.957,-40.494,-41.381,-41.266,-41.613,-41.265,-41.074,-43.344,-43.522,-42.965,-42.945,-43.533,-43.473,-45.0,-45.65,-45.79,-46.823,-48.005,-47.435,-46.653,-48.323,-48.071,-48.287,-48.718,-48.568,-49.36,-48.315,-49.668,-49.382,-49.872,-50.853,-50.495,-50.986,-51.449,-50.944,-50.879,-51.279,-51.641,-51.501,-52.067],[-28.59,-13.708,-5.514,-6.725,-6.982,-13.431,-11.822,-13.893,-18.462,-20.727,-18.159,-19.901,-22.618,-25.004,-24.799,-22.57,-20.967,-22.472,-23.611,-25.83,-26.237,-26.066,-25.499,-27.393,-29.315,-31.583,-31.195,-30.826,-32.605,-32.737,-31.754,-32.33,-31.606,-32.497,-32.632,-31.283,-32.609,-32.402,-33.189,-34.436,-34.605,-35.774,-35.323,-35.346,-36.339,-35.872,-37.579,-37.153,-38.318,-36.874,-37.369,-38.517,-39.838,-39.866,-39.497,-40.778,-40.135,-41.159,-41.577,-42.081,-42.298,-41.917,-42.359,-43.931,-44.128,-44.9,-45.161,-44.891,-45.073,-46.527,-46.844,-47.11,-47.543,-48.159,-48.113,-47.774,-49.005,-48.618,-48.599,-48.96,-49.408,-50.107,-49.274,-50.623,-49.776,-50.427,-50.999,-51.142,-51.355,-51.892,-51.761,-51.255,-51.981,-51.784,-52.08,-52.875],[-29.003,-13.288,-4.252,-7.318,-7.834,-13.025,-13.485,-14.968,-19.917,-20.579,-19.039,-20.639,-23.156,-24.448,-25.038,-21.953,-21.131,-22.817,-24.249,-26.108,-26.374,-26.171,-25.47,-28.221,-30.194,-31.204,-31.639,-31.147,-32.3,-32.101,-31.219,-31.678,-31.413,-32.214,-32.629,-31.528,-33.744,-33.784,-33.494,-35.228,-35.928,-36.389,-36.081,-35.977,-36.567,-36.329,-38.046,-37.772,-38.848,-38.118,-38.425,-39.701,-41.057,-40.842,-40.617,-41.476,-41.106,-41.884,-42.15,-42.985,-43.042,-42.844,-44.214,-44.547,-45.201,-46.412,-46.618,-45.649,-46.571,-47.73,-47.924,-47.906,-47.897,-48.079,-48.622,-48.627,-49.342,-49.266,-49.029,-49.1,-50.098,-50.303,-50.27,-51.197,-50.297,-50.868,-50.786,-52.12,-51.403,-52.168,-52.414,-51.148,-52.143,-51.918,-52.124,-53.315],[-29.6,-12.453,-3.471,-7.921,-8.876,-12.525,-15.412,-15.912,-20.166,-20.576,-19.774,-21.403,-23.737,-24.316,-24.65,-21.766,-22.118,-24.079,-25.03,-25.818,-26.673,-26.175,-25.917,-28.982,-30.768,-30.459,-31.554,-31.074,-31.901,-31.522,-30.814,-31.384,-31.722,-32.562,-33.153,-32.656,-34.689,-34.484,-34.272,-35.641,-36.726,-36.533,-36.656,-36.377,-36.557,-36.881,-38.682,-38.272,-39.296,-39.128,-39.531,-40.589,-41.419,-41.38,-41.428,-41.957,-41.725,-42.727,-43.31,-43.941,-43.829,-43.386,-44.963,-45.264,-46.36,-46.731,-47.099,-46.096,-47.339,-48.754,-49.015,-48.487,-48.447,-48.38,-49.137,-49.074,-49.431,-49.744,-49.561,-49.495,-50.479,-50.521,-51.168,-51.147,-50.869,-51.133,-50.558,-52.755,-51.395,-52.237,-52.679,-50.899,-51.88,-52.133,-51.942,-53.454],[-30.079,-11.772,-3.047,-8.427,-9.771,-12.409,-16.406,-16.349,-19.842,-20.924,-20.216,-21.788,-23.925,-24.323,-24.344,-22.187,-23.386,-25.32,-25.501,-25.854,-27.225,-26.629,-27.178,-29.813,-31.039,-30.497,-31.673,-30.751,-31.458,-31.185,-30.804,-31.346,-32.206,-33.116,-33.627,-33.856,-35.206,-34.809,-35.313,-35.9,-36.856,-36.561,-37.05,-36.884,-36.734,-37.369,-39.045,-38.81,-39.607,-39.519,-40.001,-41.052,-41.066,-41.741,-42.112,-42.628,-42.639,-43.691,-44.431,-44.848,-44.713,-44.12,-45.23,-46.238,-47.088,-46.707,-47.307,-46.933,-47.615,-49.503,-49.652,-48.787,-49.023,-49.017,-49.568,-49.441,-49.656,-50.286,-50.302,-50.157,-50.796,-51.152,-51.968,-51.151,-51.365,-51.581,-50.773,-52.883,-51.731,-52.245,-52.808,-51.348,-52.01,-52.716,-52.232,-53.598],[-30.311,-12.198,-2.665,-8.951,-10.122,-13.085,-16.719,-16.386,-19.587,-21.533,-20.315,-21.854,-23.651,-23.883,-24.167,-23.327,-24.348,-25.816,-25.548,-26.347,-27.904,-27.731,-28.718,-30.563,-31.149,-30.882,-31.6,-30.136,-30.891,-31.022,-31.027,-31.505,-32.741,-33.623,-33.524,-34.278,-35.489,-34.904,-35.412,-35.917,-36.813,-36.773,-37.289,-37.363,-37.178,-37.856,-39.187,-39.346,-39.637,-39.332,-39.807,-41.176,-40.998,-42.264,-42.55,-43.441,-43.905,-44.651,-45.027,-45.497,-45.464,-45.06,-45.664,-47.233,-47.498,-46.897,-47.588,-48.034,-47.58,-49.593,-49.545,-48.595,-49.384,-49.649,-49.979,-50.013,-50.119,-51.033,-51.104,-50.754,-51.233,-51.723,-52.421,-51.634,-51.8,-52.123,-51.39,-52.811,-52.485,-52.322,-52.949,-52.544,-52.728,-53.405,-53.006,-53.63],[-30.506,-13.896,-2.278,-9.697,-9.922,-14.374,-16.729,-16.718,-19.437,-22.078,-20.193,-21.905,-23.096,-23.051,-24.034,-24.86,-24.62,-25.707,-25.731,-27.043,-28.319,-28.979,-29.554,-30.748,-30.959,-30.71,-30.638,-29.167,-30.121,-30.982,-31.005,-31.698,-33.114,-33.789,-33.281,-34.176,-35.592,-34.69,-34.908,-35.699,-36.997,-37.155,-37.62,-37.927,-37.849,-38.496,-39.437,-39.786,-39.698,-39.531,-39.969,-41.446,-41.72,-42.911,-43.024,-44.309,-44.941,-45.438,-45.386,-46.057,-45.936,-45.91,-46.207,-47.934,-48.059,-47.427,-48.081,-48.825,-47.781,-49.385,-49.521,-48.603,-49.805,-50.112,-50.363,-50.699,-50.721,-51.778,-51.789,-51.303,-51.699,-51.899,-52.581,-52.503,-52.239,-52.654,-52.038,-52.974,-53.249,-52.688,-53.319,-53.771,-53.7,-53.799,-53.788,-53.496],[-31.019,-15.891,-2.041,-10.437,-9.645,-15.446,-16.65,-17.497,-18.881,-21.809,-20.333,-21.81,-22.421,-22.247,-23.819,-25.909,-24.499,-25.601,-26.145,-27.734,-28.345,-29.72,-29.839,-30.259,-30.368,-29.98,-29.145,-28.45,-29.714,-31.11,-30.962,-31.953,-33.43,-33.624,-33.482,-34.003,-35.505,-34.604,-34.974,-35.569,-37.366,-37.868,-38.332,-38.687,-38.691,-39.084,-39.86,-40.415,-40.346,-40.657,-41.045,-42.259,-43.167,-43.548,-43.823,-45.119,-45.562,-46.035,-45.879,-46.849,-46.459,-46.902,-47.119,-48.6,-48.762,-47.984,-48.647,-49.292,-48.624,-49.575,-50.152,-49.285,-50.506,-50.526,-50.705,-51.394,-51.295,-52.298,-52.428,-52.107,-52.163,-52.128,-52.737,-53.345,-52.813,-53.269,-52.641,-53.365,-53.764,-53.435,-53.915,-54.501,-54.542,-54.142,-54.38,-53.613],[-31.876,-16.604,-2.046,-11.035,-9.219,-15.829,-16.667,-18.267,-17.949,-20.668,-20.634,-20.768,-21.377,-21.438,-22.866,-25.637,-24.389,-25.726,-26.257,-28.291,-28.038,-29.621,-29.669,-29.447,-29.305,-28.952,-27.926,-28.3,-30.066,-31.463,-31.291,-32.509,-34.03,-33.297,-33.878,-33.803,-35.19,-34.992,-35.748,-35.494,-37.606,-39.002,-39.235,-39.282,-39.165,-39.336,-40.284,-41.293,-41.654,-42.356,-42.671,-43.199,-44.622,-44.034,-44.754,-45.516,-45.997,-46.231,-46.335,-47.543,-47.222,-47.894,-48.24,-49.256,-49.057,-47.854,-49.026,-49.542,-49.703,-50.185,-50.988,-50.242,-51.223,-50.953,-51.018,-51.984,-51.537,-52.396,-52.906,-53.017,-52.677,-52.593,-53.046,-53.631,-53.286,-53.711,-53.081,-53.505,-54.026,-54.064,-54.329,-54.859,-55.051,-54.628,-54.82,-54.193],[-32.711,-16.286,-2.268,-11.633,-8.382,-15.859,-16.772,-18.785,-17.512,-19.804,-20.363,-19.059,-20.085,-20.221,-21.227,-24.653,-24.63,-25.958,-26.258,-28.376,-27.615,-29.112,-29.007,-28.897,-27.992,-27.898,-27.387,-28.685,-30.818,-31.982,-31.583,-33.191,-34.671,-32.857,-33.958,-33.695,-34.843,-35.667,-36.548,-35.429,-37.679,-40.095,-39.533,-39.299,-38.98,-39.21,-40.482,-41.843,-42.944,-43.463,-43.726,-43.259,-45.171,-44.14,-45.312,-45.178,-46.097,-45.968,-46.343,-47.515,-47.871,-48.109,-48.616,-49.397,-48.718,-47.085,-49.268,-49.383,-50.232,-50.447,-51.22,-50.688,-51.602,-51.083,-51.146,-52.096,-51.409,-52.131,-52.943,-53.535,-53.119,-52.98,-53.468,-53.271,-53.221,-53.835,-53.12,-53.291,-54.069,-54.092,-54.25,-54.929,-55.217,-54.997,-55.003,-54.969],[-33.359,-16.446,-2.577,-12.378,-7.284,-16.059,-17.109,-19.139,-18.039,-19.732,-19.476,-17.751,-19.051,-18.827,-19.777,-23.922,-25.308,-26.172,-26.579,-27.991,-27.477,-28.854,-28.294,-28.773,-26.826,-27.138,-27.356,-29.527,-31.419,-32.499,-31.481,-33.65,-34.925,-32.598,-33.985,-33.831,-34.809,-36.251,-36.931,-35.623,-37.859,-40.753,-39.169,-39.103,-38.765,-39.062,-40.562,-41.881,-43.636,-43.624,-43.859,-42.544,-45.104,-43.929,-45.405,-44.532,-45.937,-45.771,-46.001,-47.0,-48.054,-47.716,-48.392,-49.196,-48.294,-46.5,-49.652,-48.994,-50.178,-50.158,-50.929,-50.65,-51.691,-50.88,-51.137,-51.843,-51.336,-51.969,-52.764,-53.659,-53.358,-53.199,-53.899,-52.806,-52.948,-53.96,-53.04,-53.122,-54.12,-54.025,-53.968,-54.865,-55.273,-55.205,-55.11,-55.655],[-32.433,-16.353,-2.401,-11.608,-7.846,-15.851,-16.992,-18.406,-18.242,-20.337,-19.812,-18.98,-20.013,-19.824,-21.065,-24.557,-24.907,-25.974,-26.38,-27.928,-27.8,-29.175,-28.916,-29.199,-27.875,-28.064,-27.84,-29.091,-30.745,-31.951,-31.41,-33.043,-34.373,-33.028,-33.962,-33.92,-35.097,-35.622,-36.269,-35.676,-37.708,-39.611,-39.015,-39.097,-38.878,-39.172,-40.411,-41.456,-42.349,-42.637,-42.916,-42.675,-44.522,-43.863,-44.884,-44.89,-45.893,-45.969,-46.099,-47.136,-47.508,-47.625,-48.134,-49.125,-48.611,-47.147,-49.312,-49.152,-49.699,-50.083,-50.798,-50.286,-51.299,-50.843,-51.074,-51.808,-51.416,-52.187,-52.749,-53.177,-52.899,-52.86,-53.449,-53.11,-53.054,-53.752,-53.012,-53.313,-54.051,-54.0,-54.087,-54.78,-55.06,-54.885,-54.897,-54.867],[-32.769,-16.679,-2.539,-12.009,-7.289,-16.047,-17.168,-18.513,-18.362,-20.249,-19.511,-18.494,-19.543,-19.264,-20.589,-24.314,-25.112,-26.073,-26.483,-27.86,-27.757,-29.135,-28.746,-29.084,-27.417,-27.779,-27.717,-29.379,-30.968,-32.168,-31.49,-33.334,-34.557,-32.974,-34.097,-33.987,-35.106,-35.892,-36.535,-35.794,-37.814,-40.03,-39.063,-39.188,-38.945,-39.234,-40.568,-41.666,-42.821,-43.066,-43.294,-42.648,-44.799,-43.897,-45.084,-44.816,-45.983,-46.007,-46.118,-47.149,-47.726,-47.748,-48.319,-49.228,-48.595,-47.038,-49.547,-49.076,-49.848,-50.092,-50.847,-50.447,-51.444,-50.88,-51.183,-51.876,-51.496,-52.235,-52.818,-53.417,-53.048,-53.03,-53.643,-53.069,-53.099,-53.88,-53.122,-53.335,-54.171,-54.167,-54.119,-54.882,-55.216,-55.082,-55.043,-55.194],[-32.173,-16.008,-2.432,-11.346,-7.89,-15.63,-16.98,-18.137,-18.424,-20.516,-19.786,-19.198,-20.216,-20.012,-21.312,-24.503,-24.849,-25.932,-26.313,-27.76,-27.844,-29.1,-28.927,-29.33,-28.095,-28.297,-28.093,-29.169,-30.699,-31.854,-31.374,-32.893,-34.182,-33.115,-33.929,-33.962,-35.168,-35.527,-36.117,-35.722,-37.636,-39.263,-38.82,-38.946,-38.755,-39.087,-40.314,-41.222,-41.988,-42.208,-42.51,-42.519,-44.126,-43.713,-44.61,-44.783,-45.704,-45.877,-45.995,-46.978,-47.266,-47.354,-47.868,-48.931,-48.544,-47.2,-49.171,-49.039,-49.441,-49.989,-50.646,-50.089,-51.089,-50.727,-50.99,-51.645,-51.338,-52.13,-52.6,-52.91,-52.722,-52.734,-53.333,-53.023,-52.953,-53.602,-52.886,-53.301,-53.936,-53.875,-54.007,-54.587,-54.857,-54.742,-54.729,-54.718],[-31.507,-15.127,-2.457,-10.564,-8.398,-15.02,-16.681,-17.595,-18.585,-20.751,-19.94,-19.855,-20.938,-20.827,-22.033,-24.355,-24.496,-25.713,-26.073,-27.451,-27.812,-28.77,-28.782,-29.503,-28.76,-28.845,-28.625,-29.199,-30.628,-31.634,-31.265,-32.498,-33.727,-33.198,-33.758,-33.871,-35.158,-35.185,-35.701,-35.664,-37.378,-38.486,-38.39,-38.478,-38.38,-38.752,-39.971,-40.616,-41.205,-41.284,-41.647,-42.148,-43.302,-43.304,-43.919,-44.404,-45.041,-45.436,-45.617,-46.483,-46.592,-46.612,-47.168,-48.322,-48.225,-47.182,-48.648,-48.679,-48.87,-49.747,-50.273,-49.64,-50.564,-50.399,-50.662,-51.168,-51.023,-51.796,-52.123,-52.22,-52.26,-52.343,-52.909,-52.746,-52.618,-53.162,-52.501,-53.179,-53.529,-53.49,-53.759,-54.005,-54.296,-54.286,-54.245,-54.31],[-31.674,-15.329,-2.453,-10.754,-8.25,-15.165,-16.768,-17.724,-18.56,-20.7,-19.892,-19.68,-20.74,-20.597,-21.842,-24.388,-24.589,-25.771,-26.135,-27.522,-27.826,-28.854,-28.824,-29.467,-28.584,-28.707,-28.495,-29.2,-30.645,-31.687,-31.293,-32.593,-33.836,-33.183,-33.804,-33.903,-35.17,-35.276,-35.807,-35.687,-37.446,-38.668,-38.493,-38.596,-38.473,-38.838,-40.059,-40.763,-41.388,-41.498,-41.847,-42.24,-43.492,-43.406,-44.087,-44.502,-45.208,-45.552,-45.715,-46.609,-46.757,-46.791,-47.339,-48.474,-48.311,-47.195,-48.785,-48.771,-49.008,-49.811,-50.368,-49.75,-50.692,-50.481,-50.749,-51.288,-51.105,-51.887,-52.243,-52.386,-52.374,-52.443,-53.021,-52.817,-52.704,-53.271,-52.596,-53.214,-53.632,-53.587,-53.823,-54.146,-54.432,-54.4,-54.364,-54.413],[-31.829,-15.527,-2.459,-10.937,-8.089,-15.302,-16.846,-17.835,-18.543,-20.652,-19.834,-19.506,-20.548,-20.375,-21.657,-24.403,-24.677,-25.824,-26.191,-27.579,-27.834,-28.923,-28.848,-29.428,-28.41,-28.576,-28.379,-29.216,-30.671,-31.744,-31.322,-32.692,-33.938,-33.167,-33.852,-33.931,-35.179,-35.364,-35.908,-35.713,-37.508,-38.846,-38.583,-38.701,-38.557,-38.915,-40.143,-40.9,-41.568,-41.705,-42.04,-42.317,-43.671,-43.493,-44.238,-44.58,-45.353,-45.65,-45.798,-46.715,-46.908,-46.951,-47.494,-48.607,-48.381,-47.198,-48.91,-48.842,-49.131,-49.862,-50.449,-49.851,-50.808,-50.552,-50.827,-51.393,-51.179,-51.964,-52.348,-52.541,-52.478,-52.535,-53.122,-52.875,-52.779,-53.37,-52.684,-53.245,-53.726,-53.68,-53.879,-54.273,-54.557,-54.507,-54.472,-54.516],[-30.55,-14.667,-2.605,-9.98,-8.477,-14.632,-16.329,-17.009,-18.605,-20.834,-19.916,-20.123,-21.316,-21.297,-22.424,-24.153,-24.237,-25.516,-25.87,-27.112,-27.653,-28.426,-28.502,-29.465,-29.038,-29.149,-28.977,-29.354,-30.721,-31.605,-31.257,-32.361,-33.452,-33.214,-33.675,-33.732,-35.041,-34.912,-35.378,-35.578,-37.097,-38.001,-38.015,-38.02,-38.085,-38.444,-39.69,-40.164,-40.764,-40.694,-41.092,-41.741,-42.773,-42.888,-43.333,-43.899,-44.353,-44.931,-45.181,-45.927,-46.007,-45.994,-46.55,-47.736,-47.795,-46.967,-48.088,-48.163,-48.332,-49.35,-49.804,-49.231,-50.09,-50.059,-50.287,-50.646,-50.671,-51.346,-51.605,-51.644,-51.776,-51.928,-52.385,-52.367,-52.22,-52.724,-52.18,-52.952,-53.142,-53.181,-53.476,-53.495,-53.843,-53.914,-53.848,-54.033],[-30.778,-14.777,-2.577,-10.117,-8.431,-14.722,-16.41,-17.138,-18.606,-20.811,-19.909,-20.037,-21.203,-21.155,-22.31,-24.188,-24.297,-25.561,-25.917,-27.185,-27.687,-28.499,-28.557,-29.468,-28.951,-29.066,-28.888,-29.329,-30.707,-31.619,-31.262,-32.4,-33.517,-33.208,-33.699,-33.764,-35.066,-34.98,-35.456,-35.601,-37.162,-38.116,-38.098,-38.122,-38.152,-38.513,-39.757,-40.268,-40.871,-40.83,-41.22,-41.829,-42.895,-42.979,-43.463,-44.006,-44.5,-45.04,-45.276,-46.047,-46.138,-46.129,-46.687,-47.864,-47.888,-47.012,-48.216,-48.272,-48.453,-49.438,-49.909,-49.327,-50.198,-50.137,-50.375,-50.763,-50.754,-51.449,-51.722,-51.778,-51.889,-52.027,-52.508,-52.453,-52.311,-52.826,-52.257,-53.007,-53.233,-53.258,-53.543,-53.61,-53.949,-54.003,-53.941,-54.106],[-30.977,-14.888,-2.554,-10.248,-8.376,-14.811,-16.485,-17.256,-18.605,-20.788,-19.897,-19.949,-21.09,-21.017,-22.198,-24.219,-24.356,-25.603,-25.962,-27.25,-27.715,-28.567,-28.605,-29.466,-28.862,-28.984,-28.804,-29.31,-30.699,-31.636,-31.269,-32.443,-33.581,-33.202,-33.723,-33.794,-35.089,-35.044,-35.531,-35.623,-37.221,-38.229,-38.176,-38.217,-38.216,-38.579,-39.821,-40.367,-40.978,-40.964,-41.346,-41.91,-43.014,-43.064,-43.587,-44.103,-44.638,-45.141,-45.363,-46.158,-46.261,-46.257,-46.816,-47.984,-47.973,-47.05,-48.334,-48.369,-48.565,-49.515,-50.004,-49.416,-50.299,-50.207,-50.455,-50.87,-50.829,-51.541,-51.828,-51.903,-51.991,-52.117,-52.618,-52.529,-52.394,-52.92,-52.328,-53.054,-53.317,-53.33,-53.602,-53.718,-54.048,-54.087,-54.028,-54.176],[-28.212,-14.215,-2.918,-9.179,-8.373,-14.23,-15.857,-16.048,-18.415,-20.897,-19.852,-20.357,-21.729,-21.905,-22.924,-24.038,-24.056,-25.335,-25.656,-26.64,-27.376,-28.063,-28.203,-29.344,-29.271,-29.499,-29.394,-29.623,-30.931,-31.675,-31.344,-32.342,-33.22,-33.267,-33.643,-33.57,-34.874,-34.575,-34.97,-35.444,-36.707,-37.468,-37.574,-37.44,-37.786,-38.105,-39.318,-39.651,-40.3,-40.074,-40.5,-41.195,-42.194,-42.33,-42.617,-43.186,-43.487,-44.267,-44.595,-45.184,-45.268,-45.298,-45.756,-46.971,-47.155,-46.551,-47.229,-47.4,-47.559,-48.623,-49.005,-48.565,-49.385,-49.492,-49.596,-49.806,-49.989,-50.487,-50.721,-50.717,-50.829,-51.067,-51.324,-51.563,-51.449,-51.903,-51.615,-52.34,-52.433,-52.595,-52.895,-52.707,-53.139,-53.285,-53.238,-53.498]],"outside":[[-19.344,-19.123,-7.826,-11.246,-7.131,-19.591,-24.845,-7.034,-19.125,-20.57,-16.524,-17.881,-13.194,-18.845,-23.619,-9.881,-18.015,-19.529,-21.248,-23.259,-18.943,-23.876,-27.646,-19.568,-25.022,-26.811,-27.449,-27.931,-26.913,-29.281,-30.093,-27.164,-28.957,-30.317,-30.686,-30.175,-28.818,-31.022,-32.866,-29.304,-32.316,-30.638,-32.287,-32.461,-29.785,-32.546,-31.39,-32.66,-34.332,-32.067,-35.354,-34.983,-34.551,-36.129,-33.661,-35.742,-36.539,-34.15,-37.329,-37.877,-38.46,-38.131,-38.75,-39.181,-39.25,-39.395,-39.571,-40.89,-41.377,-41.82,-41.727,-42.922,-42.996,-43.11,-44.018,-43.568,-43.295,-43.205,-43.097,-44.269,-45.705,-44.862,-44.712,-45.335,-46.718,-46.996,-45.273,-47.61,-48.073,-47.764,-47.565,-47.755,-48.294,-48.805,-47.38,-49.215],[-19.179,-19.132,-7.84,-11.432,-7.422,-19.617,-24.771,-6.931,-19.024,-20.404,-16.4,-17.767,-13.116,-18.637,-23.446,-9.635,-17.974,-19.41,-21.147,-23.174,-18.899,-23.728,-27.534,-19.373,-24.97,-26.698,-27.414,-27.874,-26.891,-29.117,-30.099,-27.026,-28.982,-30.259,-30.718,-30.205,-28.84,-30.93,-32.863,-29.208,-32.298,-30.669,-32.295,-32.474,-29.845,-32.588,-31.49,-32.669,-34.348,-32.142,-35.4,-35.039,-34.628,-36.19,-33.757,-35.804,-36.61,-34.255,-37.406,-37.934,-38.535,-38.2,-38.834,-39.254,-39.331,-39.475,-39.661,-40.978,-41.467,-41.915,-41.818,-43.011,-43.087,-43.186,-44.096,-43.669,-43.397,-43.296,-43.213,-44.38,-45.764,-44.946,-44.788,-45.44,-46.798,-47.058,-45.378,-47.715,-48.147,-47.857,-47.671,-47.865,-48.406,-48.905,-47.493,-49.321],[-19.024,-19.147,-7.862,-11.627,-7.723,-19.647,-24.698,-6.829,-18.93,-20.247,-16.282,-17.661,-13.047,-18.437,-23.278,-9.398,-17.944,-19.3,-21.045,-23.088,-18.862,-23.587,-27.426,-19.183,-24.925,-26.594,-27.386,-27.824,-26.877,-28.957,-30.111,-26.891,-29.01,-30.207,-30.752,-30.241,-28.869,-30.844,-32.869,-29.122,-32.287,-30.71,-32.313,-32.497,-29.919,-32.64,-31.604,-32.689,-34.373,-32.229,-35.457,-35.106,-34.718,-36.263,-33.87,-35.879,-36.696,-34.376,-37.499,-38.005,-38.623,-38.284,-38.932,-39.341,-39.426,-39.569,-39.767,-41.079,-41.572,-42.023,-41.922,-43.111,-43.19,-43.275,-44.185,-43.781,-43.512,-43.399,-43.342,-44.503,-45.833,-45.042,-44.877,-45.557,-46.89,-47.131,-45.499,-47.832,-48.232,-47.962,-47.789,-47.986,-48.529,-49.014,-47.62,-49.439],[-18.878,-19.167,-7.891,-11.831,-8.033,-19.681,-24.629,-6.73,-18.844,-20.1,-16.172,-17.564,-12.987,-18.246,-23.116,-9.171,-17.923,-19.199,-20.941,-23.0,-18.831,-23.454,-27.321,-18.997,-24.887,-26.5,-27.365,-27.781,-26.869,-28.803,-30.129,-26.759,-29.042,-30.161,-30.789,-30.283,-28.905,-30.766,-32.883,-29.047,-32.282,-30.763,-32.342,-32.531,-30.007,-32.702,-31.733,-32.719,-34.407,-32.328,-35.525,-35.183,-34.823,-36.346,-34.0,-35.966,-36.795,-34.514,-37.607,-38.09,-38.725,-38.383,-39.047,-39.443,-39.534,-39.678,-39.889,-41.195,-41.691,-42.144,-42.04,-43.222,-43.304,-43.375,-44.284,-43.904,-43.639,-43.514,-43.482,-44.636,-45.912,-45.148,-44.978,-45.686,-46.993,-47.215,-45.633,-47.959,-48.326,-48.076,-47.918,-48.118,-48.663,-49.134,-47.761,-49.568],[-20.333,-19.065,-7.744,-10.405,-5.948,-19.471,-25.228,-7.566,-19.687,-21.528,-17.22,-18.504,-13.595,-20.069,-24.614,-11.376,-18.2,-20.186,-21.747,-23.685,-19.161,-24.703,-28.211,-20.665,-25.257,-27.419,-27.599,-28.217,-26.963,-30.187,-30.011,-27.877,-28.796,-30.579,-30.481,-29.971,-28.669,-31.517,-32.847,-29.83,-32.391,-30.446,-32.217,-32.366,-29.439,-32.301,-30.857,-32.59,-34.232,-31.664,-35.098,-34.689,-34.132,-35.806,-33.14,-35.403,-36.144,-33.592,-36.905,-37.57,-38.061,-37.75,-38.301,-38.783,-38.819,-38.966,-39.087,-40.418,-40.888,-41.311,-41.24,-42.447,-42.507,-42.69,-43.591,-43.033,-42.75,-42.716,-42.496,-43.694,-45.374,-44.408,-44.304,-44.785,-46.283,-46.654,-44.711,-47.065,-47.668,-47.271,-47.018,-47.19,-47.717,-48.295,-46.78,-48.658],[-19.962,-19.036,-7.707,-10.665,-6.397,-19.478,-25.072,-7.372,-19.451,-21.147,-16.945,-18.236,-13.387,-19.634,-24.275,-10.859,-18.041,-19.903,-21.593,-23.56,-19.027,-24.38,-27.995,-20.296,-25.105,-27.135,-27.471,-28.05,-26.859,-29.859,-29.969,-27.618,-28.806,-30.43,-30.508,-29.968,-28.644,-31.298,-32.771,-29.567,-32.309,-30.409,-32.138,-32.298,-29.426,-32.289,-30.915,-32.52,-34.182,-31.697,-35.083,-34.697,-34.15,-35.824,-33.171,-35.405,-36.157,-33.645,-36.913,-37.558,-38.083,-37.747,-38.319,-38.795,-38.846,-38.987,-39.116,-40.459,-40.933,-41.371,-41.291,-42.511,-42.575,-42.726,-43.638,-43.119,-42.829,-42.774,-42.606,-43.8,-45.399,-44.467,-44.332,-44.87,-46.334,-46.67,-44.783,-47.161,-47.71,-47.344,-47.107,-47.288,-47.819,-48.382,-46.866,-48.751],[-22.145,-19.202,-7.944,-9.522,-4.523,-19.495,-25.939,-8.404,-20.782,-23.464,-18.506,-19.783,-14.6,-22.25,-26.173,-14.097,-19.007,-21.543,-22.339,-24.163,-19.808,-26.27,-29.167,-22.367,-25.976,-28.868,-28.224,-29.023,-27.503,-31.74,-30.23,-29.031,-28.796,-31.259,-30.391,-30.035,-28.823,-32.538,-33.259,-31.164,-32.79,-30.702,-32.657,-32.754,-29.626,-32.434,-30.742,-32.989,-34.513,-31.623,-35.269,-34.767,-34.194,-35.85,-33.171,-35.521,-36.218,-33.51,-37.017,-37.764,-38.103,-37.915,-38.377,-38.881,-38.848,-39.023,-39.117,-40.376,-40.833,-41.18,-41.147,-42.295,-42.327,-42.652,-43.496,-42.769,-42.526,-42.586,-42.141,-43.35,-45.358,-44.255,-44.304,-44.542,-46.174,-46.697,-44.532,-46.779,-47.591,-47.072,-46.775,-46.903,-47.415,-48.056,-46.543,-48.383],[-21.528,-19.056,-7.757,-9.772,-5.101,-19.458,-25.637,-8.045,-20.343,-22.662,-17.992,-19.22,-14.093,-21.418,-25.638,-13.08,-18.507,-20.945,-22.095,-23.991,-19.468,-25.617,-28.714,-21.702,-25.537,-28.188,-27.824,-28.589,-27.074,-31.094,-29.962,-28.511,-28.676,-30.851,-30.282,-29.814,-28.587,-32.079,-32.933,-30.542,-32.543,-30.396,-32.285,-32.412,-29.276,-32.178,-30.513,-32.667,-34.251,-31.418,-35.004,-34.574,-33.911,-35.663,-32.852,-35.249,-35.938,-33.245,-36.697,-37.487,-37.868,-37.589,-38.077,-38.599,-38.598,-38.75,-38.826,-40.139,-40.597,-40.978,-40.936,-42.136,-42.169,-42.431,-43.307,-42.638,-42.364,-42.385,-42.047,-43.264,-45.17,-44.094,-44.076,-44.398,-45.999,-46.469,-44.333,-46.681,-47.403,-46.929,-46.65,-46.796,-47.308,-47.954,-46.358,-48.258],[-23.295,-19.31,-8.157,-9.115,-3.842,-19.567,-26.485,-8.961,-21.522,-25.092,-19.44,-20.795,-15.497,-23.934,-27.171,-16.363,-19.889,-22.579,-22.633,-24.434,-20.395,-27.517,-29.903,-23.556,-26.665,-30.248,-28.866,-29.789,-28.192,-32.967,-30.541,-29.885,-28.902,-31.867,-30.419,-30.266,-29.069,-33.297,-33.768,-32.357,-33.172,-31.118,-33.24,-33.269,-30.141,-32.753,-30.989,-33.493,-34.853,-31.816,-35.641,-35.056,-34.609,-36.165,-33.635,-35.916,-36.598,-33.822,-37.495,-38.216,-38.444,-38.425,-38.828,-39.319,-39.21,-39.418,-39.528,-40.661,-41.124,-41.388,-41.388,-42.462,-42.467,-42.924,-43.691,-42.819,-42.642,-42.794,-42.114,-43.319,-45.573,-44.367,-44.598,-44.638,-46.356,-47.007,-44.728,-46.838,-47.804,-47.187,-46.897,-46.984,-47.493,-48.176,-46.702,-48.464],[-23.802,-19.489,-8.508,-8.914,-3.24,-19.525,-27.035,-9.495,-22.087,-27.036,-20.402,-22.138,-16.772,-25.697,-27.931,-18.845,-21.545,-23.869,-22.911,-24.687,-21.114,-28.961,-30.979,-25.039,-27.952,-32.055,-29.961,-30.97,-29.724,-34.524,-31.347,-31.073,-29.239,-32.94,-30.735,-31.037,-29.752,-34.113,-34.702,-33.739,-33.723,-32.067,-34.46,-34.275,-31.614,-33.594,-32.015,-34.437,-35.501,-32.551,-36.497,-35.666,-35.83,-36.962,-35.151,-36.951,-37.709,-35.051,-38.853,-39.067,-39.303,-39.702,-40.065,-40.43,-40.206,-40.491,-40.77,-41.635,-42.069,-42.263,-42.294,-43.146,-43.195,-43.848,-44.459,-43.444,-43.402,-43.741,-42.623,-43.777,-46.29,-45.031,-45.46,-45.365,-47.101,-47.857,-45.69,-47.408,-48.596,-47.865,-47.601,-47.642,-48.198,-48.87,-47.681,-49.177],[-23.517,-19.048,-8.254,-8.868,-3.328,-19.397,-27.249,-9.296,-21.914,-27.501,-20.663,-22.696,-17.054,-25.951,-28.21,-19.556,-22.288,-23.989,-22.568,-24.765,-21.259,-29.399,-31.184,-25.419,-28.343,-32.642,-30.193,-31.462,-30.345,-35.043,-31.285,-31.205,-29.144,-32.989,-30.471,-31.036,-29.654,-34.011,-34.924,-33.963,-33.792,-32.17,-34.898,-34.441,-32.243,-33.66,-32.246,-34.765,-35.384,-32.521,-36.659,-35.828,-36.377,-37.504,-35.906,-37.331,-38.052,-35.49,-39.436,-39.191,-39.513,-40.155,-40.515,-40.863,-40.538,-40.827,-41.189,-41.863,-42.202,-42.407,-42.538,-43.246,-43.327,-44.11,-44.562,-43.4,-43.484,-44.063,-42.469,-43.592,-46.42,-45.038,-45.629,-45.492,-47.25,-48.061,-45.956,-47.532,-48.798,-48.019,-47.903,-47.921,-48.516,-49.319,-48.047,-49.369],[-23.394,-18.378,-7.343,-8.758,-4.845,-20.042,-26.477,-8.492,-21.24,-24.405,-19.309,-20.011,-14.777,-23.417,-27.364,-16.144,-18.862,-21.378,-21.799,-24.652,-19.882,-26.865,-28.836,-22.265,-25.229,-30.002,-27.843,-29.276,-26.785,-32.231,-28.913,-28.669,-28.116,-30.532,-29.209,-28.75,-27.454,-32.547,-32.952,-31.658,-32.65,-29.617,-31.998,-32.099,-28.56,-31.487,-29.315,-32.413,-33.696,-30.109,-34.639,-34.452,-33.292,-35.49,-32.069,-34.657,-35.147,-32.006,-36.126,-37.486,-37.437,-37.264,-37.678,-38.366,-38.073,-38.236,-38.027,-39.133,-39.788,-39.855,-39.91,-41.31,-41.074,-41.634,-42.357,-41.125,-41.003,-41.173,-40.368,-41.687,-44.298,-42.749,-43.394,-43.023,-45.034,-45.896,-43.041,-45.638,-46.518,-45.789,-45.697,-45.732,-46.117,-47.079,-44.899,-46.917],[-23.186,-18.475,-6.896,-9.168,-6.884,-20.037,-25.369,-7.732,-20.466,-22.3,-17.598,-18.089,-12.946,-21.478,-26.009,-13.593,-16.62,-19.971,-21.73,-23.713,-18.884,-25.115,-27.481,-20.807,-23.687,-27.575,-26.287,-27.254,-24.878,-30.315,-28.121,-27.282,-27.492,-29.219,-28.872,-27.795,-26.816,-31.531,-31.403,-30.121,-31.648,-28.565,-30.337,-30.741,-26.719,-30.257,-28.0,-31.086,-32.896,-29.392,-33.222,-33.274,-31.542,-33.996,-30.098,-33.195,-33.66,-30.524,-34.246,-36.175,-36.119,-35.436,-35.794,-36.551,-36.578,-36.653,-36.449,-37.865,-38.418,-38.681,-38.676,-40.203,-40.005,-40.203,-41.201,-40.428,-40.018,-39.908,-39.913,-41.273,-43.296,-41.954,-42.091,-42.085,-43.93,-44.585,-41.787,-44.688,-45.307,-44.806,-44.525,-44.63,-45.02,-45.893,-43.642,-45.964],[-21.593,-18.559,-6.843,-10.642,-8.782,-19.053,-24.49,-6.87,-19.455,-20.924,-16.453,-17.545,-12.045,-19.899,-24.699,-11.676,-16.034,-19.506,-21.631,-23.005,-18.079,-24.001,-27.052,-20.342,-23.672,-25.931,-25.891,-26.458,-24.684,-29.321,-28.556,-26.934,-27.71,-29.05,-29.521,-28.261,-27.454,-30.782,-30.976,-28.928,-31.264,-28.747,-30.21,-30.52,-26.776,-30.365,-28.486,-31.046,-32.938,-30.154,-33.107,-32.978,-31.566,-33.858,-30.193,-33.268,-33.832,-31.073,-34.193,-35.47,-35.885,-35.087,-35.538,-36.192,-36.453,-36.481,-36.472,-38.225,-38.477,-39.134,-39.093,-40.42,-40.554,-40.429,-41.564,-41.414,-40.799,-40.569,-41.091,-42.342,-43.624,-42.824,-42.15,-42.967,-44.353,-44.557,-42.463,-45.279,-45.663,-45.477,-45.035,-45.314,-45.831,-46.46,-44.581,-46.829],[-19.021,-18.614,-7.069,-13.004,-10.545,-18.353,-23.926,-6.264,-18.284,-19.677,-15.884,-17.357,-11.922,-18.511,-23.329,-9.807,-16.54,-19.0,-21.26,-23.114,-17.361,-22.924,-26.974,-19.578,-24.19,-24.965,-26.199,-26.744,-25.556,-28.568,-29.507,-26.803,-28.631,-29.591,-30.772,-29.817,-28.399,-30.042,-31.468,-27.546,-31.378,-29.348,-30.969,-30.986,-28.36,-31.62,-30.618,-31.322,-33.404,-31.438,-34.099,-33.497,-33.117,-34.844,-32.166,-34.323,-35.267,-33.244,-35.718,-35.621,-36.706,-36.153,-36.939,-37.332,-37.594,-37.669,-37.9,-39.771,-39.998,-40.996,-40.762,-41.95,-42.512,-42.191,-43.276,-43.575,-42.987,-42.746,-43.404,-44.447,-44.948,-44.728,-43.404,-45.205,-46.021,-45.679,-44.762,-47.275,-47.334,-47.373,-46.987,-47.465,-48.222,-48.388,-47.291,-49.127],[-17.098,-18.991,-7.451,-14.848,-12.163,-18.741,-23.664,-6.226,-17.444,-18.707,-15.373,-16.596,-12.116,-17.375,-22.343,-8.254,-17.377,-18.062,-20.938,-23.477,-17.673,-22.269,-27.123,-18.68,-24.495,-24.844,-26.741,-27.225,-26.939,-27.978,-30.509,-26.368,-29.518,-30.057,-31.838,-31.52,-29.092,-29.656,-32.538,-26.974,-31.688,-30.337,-31.934,-31.964,-30.972,-33.273,-33.891,-31.789,-33.895,-32.721,-35.502,-34.765,-35.793,-36.494,-35.786,-36.022,-37.394,-36.458,-38.421,-37.26,-38.832,-38.669,-39.905,-39.936,-40.022,-40.246,-40.851,-42.106,-42.925,-43.856,-43.269,-44.583,-45.119,-44.772,-45.723,-46.023,-45.915,-45.608,-45.991,-46.85,-46.81,-46.517,-45.61,-47.786,-48.336,-47.794,-47.999,-49.986,-49.638,-49.706,-49.77,-50.227,-51.166,-50.795,-50.7,-51.959],[-16.453,-19.313,-7.944,-15.42,-13.293,-19.848,-23.767,-6.301,-17.271,-18.272,-14.743,-16.01,-12.212,-16.553,-21.802,-7.349,-17.927,-17.49,-20.708,-23.152,-18.691,-22.093,-27.19,-18.033,-24.671,-25.352,-27.354,-27.296,-28.175,-27.783,-31.234,-26.037,-29.898,-30.04,-32.293,-32.189,-29.665,-29.697,-33.593,-27.605,-32.032,-31.771,-32.85,-33.191,-32.999,-34.23,-36.118,-32.898,-34.434,-34.103,-36.729,-36.51,-38.264,-38.041,-39.447,-38.098,-39.416,-39.277,-41.136,-40.12,-41.554,-41.717,-43.167,-42.846,-42.846,-43.15,-44.35,-44.534,-45.907,-46.161,-45.632,-46.923,-46.759,-46.51,-47.477,-47.595,-47.981,-47.391,-47.379,-48.204,-47.948,-47.429,-47.797,-49.41,-49.995,-49.82,-50.335,-51.754,-51.198,-51.298,-52.02,-52.223,-52.94,-52.494,-52.899,-53.699],[-16.536,-19.375,-8.401,-15.808,-14.432,-20.891,-24.278,-6.142,-17.535,-18.212,-14.383,-16.061,-12.393,-16.049,-21.217,-6.832,-18.237,-17.585,-20.428,-22.304,-19.237,-22.078,-26.838,-17.384,-24.937,-26.045,-28.112,-27.466,-28.663,-27.752,-31.641,-26.037,-30.275,-30.048,-32.424,-32.19,-30.209,-29.841,-34.261,-28.748,-32.541,-33.237,-33.885,-34.301,-33.885,-34.731,-36.859,-34.338,-35.353,-35.569,-38.018,-38.255,-39.816,-39.327,-41.754,-40.053,-41.161,-41.202,-43.205,-43.111,-43.71,-44.108,-45.445,-44.801,-45.082,-45.265,-46.912,-46.54,-47.663,-47.28,-47.159,-47.982,-47.513,-47.319,-48.226,-48.224,-48.494,-47.735,-47.616,-48.559,-48.35,-48.234,-49.36,-50.244,-50.642,-50.79,-51.144,-52.4,-51.895,-52.153,-53.187,-53.193,-53.596,-53.42,-53.82,-54.259],[-16.86,-19.513,-8.923,-16.569,-16.047,-21.52,-25.012,-5.793,-18.001,-18.433,-14.534,-16.543,-12.79,-15.806,-20.519,-6.37,-18.619,-18.174,-19.926,-21.495,-19.29,-22.054,-26.08,-16.591,-25.259,-26.669,-28.721,-27.966,-28.376,-27.426,-31.659,-25.871,-30.879,-30.253,-32.246,-31.908,-30.719,-30.041,-34.648,-29.693,-33.319,-34.494,-35.004,-35.286,-34.679,-35.297,-37.539,-35.75,-36.813,-37.1,-39.494,-39.79,-40.957,-40.5,-43.104,-41.312,-42.717,-42.719,-44.767,-45.214,-45.14,-45.482,-46.879,-45.921,-46.53,-46.743,-48.302,-47.833,-48.421,-47.835,-48.059,-48.331,-48.07,-47.914,-48.306,-48.118,-48.182,-47.665,-47.782,-48.89,-48.887,-49.339,-50.279,-50.675,-50.843,-51.06,-51.435,-52.712,-52.293,-52.737,-53.438,-53.414,-53.765,-53.893,-54.175,-54.301],[-17.585,-20.299,-10.199,-18.029,-18.073,-21.75,-25.154,-5.178,-18.672,-18.996,-15.277,-17.237,-13.203,-15.662,-20.253,-5.892,-19.32,-19.103,-18.995,-20.919,-19.548,-22.011,-25.227,-15.785,-25.537,-26.962,-28.613,-28.645,-27.846,-26.471,-31.155,-24.857,-31.145,-30.212,-31.454,-31.429,-31.133,-30.393,-34.799,-30.049,-33.984,-35.23,-35.827,-36.176,-35.745,-35.969,-38.54,-36.666,-38.461,-38.619,-40.698,-40.905,-41.767,-41.417,-43.677,-41.649,-43.679,-43.847,-45.728,-45.974,-45.898,-45.838,-47.644,-46.349,-46.882,-47.466,-48.815,-48.238,-48.542,-48.005,-48.511,-48.101,-47.948,-48.058,-47.751,-47.652,-47.985,-47.956,-48.571,-49.514,-49.526,-50.213,-50.478,-50.788,-51.103,-51.312,-51.886,-52.796,-52.323,-52.89,-52.952,-53.049,-53.541,-53.859,-53.969,-54.108],[-19.366,-22.221,-12.64,-20.282,-19.375,-21.825,-24.075,-4.43,-19.699,-19.715,-16.405,-18.091,-13.73,-15.493,-20.883,-5.622,-20.213,-19.993,-17.763,-20.325,-19.808,-21.973,-24.731,-15.238,-25.587,-26.648,-27.959,-29.065,-27.109,-25.439,-30.372,-23.616,-30.189,-29.66,-30.378,-30.904,-31.152,-30.852,-34.638,-30.038,-33.965,-34.876,-36.063,-36.509,-36.285,-36.502,-39.206,-36.659,-39.296,-39.673,-41.175,-41.007,-41.354,-41.978,-43.29,-41.668,-43.645,-44.369,-45.867,-45.722,-45.81,-45.735,-47.49,-46.012,-46.345,-47.187,-48.617,-48.114,-48.265,-47.726,-48.19,-46.992,-46.941,-47.563,-47.413,-47.731,-48.447,-48.619,-49.653,-49.895,-49.601,-50.356,-50.577,-50.927,-51.578,-51.557,-52.165,-52.386,-51.849,-52.453,-52.426,-52.557,-53.027,-53.347,-53.045,-53.748],[-21.662,-24.353,-14.964,-22.185,-19.577,-22.038,-22.615,-3.989,-20.591,-19.912,-16.985,-18.833,-14.46,-15.423,-21.83,-5.617,-20.853,-20.253,-16.874,-19.764,-19.456,-21.866,-24.735,-15.081,-25.121,-25.878,-27.354,-28.863,-25.99,-25.133,-29.783,-22.987,-28.697,-29.27,-29.845,-30.504,-30.7,-31.32,-34.439,-30.096,-33.656,-34.081,-35.963,-36.354,-36.227,-37.009,-39.359,-36.509,-39.161,-39.971,-41.282,-40.594,-40.222,-42.353,-42.669,-41.849,-43.178,-44.396,-45.633,-45.205,-45.444,-45.781,-46.942,-45.65,-45.979,-46.812,-48.194,-47.867,-47.638,-47.01,-47.35,-45.925,-46.318,-47.332,-47.869,-48.32,-49.032,-48.887,-50.032,-49.733,-49.459,-50.36,-51.085,-51.153,-51.79,-51.377,-51.869,-51.778,-51.365,-51.962,-52.311,-52.312,-52.555,-52.787,-51.957,-53.28],[-22.649,-24.723,-15.926,-22.528,-19.709,-22.526,-21.314,-3.853,-20.806,-19.376,-16.603,-19.012,-14.909,-15.617,-22.203,-5.706,-21.05,-20.185,-16.656,-19.696,-19.262,-21.776,-24.887,-15.189,-24.231,-25.007,-26.879,-28.47,-25.067,-25.535,-29.622,-22.739,-27.571,-29.232,-29.788,-30.322,-30.362,-31.855,-34.376,-30.473,-33.348,-33.635,-35.906,-36.198,-36.329,-37.601,-39.557,-36.586,-38.712,-39.833,-41.335,-40.438,-39.706,-42.487,-42.443,-42.011,-42.682,-44.091,-45.555,-44.773,-45.008,-45.749,-46.465,-45.601,-45.901,-46.557,-47.663,-47.321,-46.435,-46.028,-46.598,-45.663,-46.554,-47.511,-48.532,-48.667,-49.193,-48.581,-49.783,-49.606,-49.649,-50.395,-51.488,-51.041,-51.53,-50.953,-51.46,-51.386,-51.148,-51.561,-52.293,-52.084,-52.206,-52.444,-51.4,-53.069],[-22.408,-23.582,-16.08,-21.912,-19.765,-23.135,-20.515,-3.8,-20.576,-18.326,-16.098,-18.15,-15.021,-16.084,-21.548,-5.845,-20.948,-20.327,-17.1,-20.462,-19.76,-21.926,-24.861,-15.484,-23.283,-24.261,-26.592,-28.118,-24.522,-26.159,-29.824,-22.291,-26.712,-29.044,-29.799,-30.432,-30.34,-32.451,-34.227,-31.296,-33.104,-33.693,-36.112,-36.245,-36.636,-38.211,-39.738,-36.818,-38.347,-39.624,-41.535,-40.695,-39.863,-42.472,-42.543,-42.317,-42.331,-43.688,-45.805,-44.487,-44.305,-45.558,-45.938,-45.646,-45.723,-46.169,-46.851,-46.337,-44.833,-45.282,-46.253,-46.206,-47.272,-47.616,-48.878,-48.577,-48.763,-48.217,-49.396,-49.777,-49.915,-50.155,-51.41,-50.598,-50.965,-50.612,-51.12,-51.196,-51.009,-51.037,-52.154,-51.785,-52.008,-52.466,-51.681,-53.373],[-22.262,-22.315,-16.05,-21.464,-19.407,-23.442,-20.58,-3.763,-20.478,-17.231,-16.086,-16.498,-14.789,-16.563,-20.068,-6.081,-20.771,-20.806,-17.935,-22.043,-19.952,-22.462,-24.334,-15.869,-22.541,-23.916,-26.564,-27.374,-23.786,-26.375,-29.801,-21.511,-25.651,-28.559,-29.795,-30.647,-30.143,-33.034,-33.638,-32.378,-33.194,-34.333,-36.617,-36.543,-36.438,-39.019,-39.402,-37.343,-38.388,-39.434,-41.96,-41.212,-39.507,-42.48,-42.277,-43.04,-42.196,-43.392,-46.304,-44.356,-43.265,-45.55,-45.064,-45.463,-45.357,-45.601,-45.693,-45.247,-43.565,-45.354,-46.119,-47.002,-47.874,-47.197,-48.679,-48.008,-47.677,-48.232,-48.818,-49.773,-49.669,-49.557,-50.872,-50.091,-50.156,-50.402,-50.515,-51.029,-50.693,-50.341,-52.09,-51.611,-51.886,-52.743,-52.444,-53.914],[-22.823,-21.913,-16.277,-21.947,-18.788,-23.571,-21.487,-3.666,-20.903,-16.786,-16.591,-15.534,-14.245,-16.844,-18.767,-6.411,-20.438,-21.488,-18.634,-23.682,-18.693,-23.208,-23.392,-16.104,-22.214,-24.178,-26.587,-26.299,-22.828,-26.28,-29.004,-21.177,-24.583,-28.399,-30.131,-30.771,-29.816,-33.748,-33.068,-33.401,-33.788,-35.637,-37.464,-37.189,-35.511,-40.257,-38.791,-38.296,-38.822,-39.431,-42.496,-41.759,-38.466,-42.635,-41.693,-43.952,-42.042,-43.449,-46.873,-44.632,-42.11,-45.815,-44.313,-45.299,-44.844,-45.075,-44.858,-44.865,-43.289,-46.317,-45.892,-47.48,-47.977,-46.558,-48.228,-47.499,-46.834,-48.615,-48.044,-49.372,-48.95,-48.91,-50.071,-49.66,-49.594,-50.329,-49.961,-51.138,-50.477,-49.714,-52.415,-51.757,-51.852,-53.072,-53.076,-54.446],[-23.224,-22.413,-16.488,-22.415,-18.579,-23.823,-21.724,-3.531,-21.012,-17.221,-16.868,-16.285,-14.344,-16.911,-18.441,-6.433,-20.461,-21.564,-18.527,-23.193,-18.492,-23.16,-23.551,-16.119,-22.6,-24.414,-26.652,-26.136,-22.987,-26.5,-28.81,-21.742,-24.796,-28.785,-30.45,-30.947,-30.076,-33.941,-33.529,-33.499,-34.039,-35.824,-37.743,-37.441,-35.506,-40.36,-39.049,-38.427,-38.935,-39.869,-42.68,-41.838,-38.844,-42.921,-42.026,-43.867,-41.967,-43.754,-46.998,-44.915,-42.138,-45.882,-44.73,-45.569,-44.675,-45.273,-45.421,-45.461,-43.867,-46.527,-46.082,-47.49,-47.843,-46.83,-48.459,-47.949,-47.284,-48.79,-48.203,-49.537,-49.026,-49.051,-50.173,-49.709,-50.001,-50.627,-50.391,-51.549,-50.643,-49.746,-52.694,-52.095,-52.22,-53.329,-53.259,-54.674],[-23.026,-22.745,-16.317,-22.362,-18.748,-23.617,-21.563,-3.578,-20.897,-17.584,-16.771,-16.821,-14.489,-16.685,-18.811,-6.289,-20.59,-21.239,-18.133,-22.267,-18.752,-22.83,-23.878,-15.959,-22.906,-24.483,-26.704,-26.449,-23.394,-26.386,-29.001,-22.001,-25.302,-28.928,-30.346,-30.873,-30.195,-33.486,-33.784,-32.797,-33.883,-35.29,-37.343,-37.179,-35.746,-39.666,-39.236,-37.995,-38.872,-39.926,-42.411,-41.552,-39.183,-42.868,-42.226,-43.423,-42.095,-43.833,-46.711,-44.883,-42.617,-45.842,-45.103,-45.625,-44.883,-45.539,-45.884,-45.811,-44.277,-46.34,-46.225,-47.135,-47.626,-47.05,-48.553,-48.159,-47.67,-48.731,-48.526,-49.626,-49.22,-49.329,-50.463,-49.919,-50.318,-50.734,-50.674,-51.572,-50.73,-50.05,-52.623,-52.121,-52.285,-53.196,-52.909,-54.372],[-22.207,-22.899,-15.429,-21.868,-18.936,-23.14,-21.613,-3.747,-20.643,-18.041,-16.585,-17.278,-14.491,-16.318,-19.599,-6.091,-20.612,-20.769,-17.771,-21.4,-19.051,-22.469,-24.225,-15.736,-23.338,-24.71,-26.851,-27.072,-24.034,-26.139,-29.334,-22.283,-26.079,-29.046,-30.213,-30.763,-30.318,-32.71,-33.983,-31.833,-33.697,-34.757,-36.775,-36.799,-35.941,-38.647,-39.277,-37.429,-38.768,-39.761,-41.936,-41.176,-39.516,-42.611,-42.379,-42.837,-42.334,-43.853,-46.285,-44.868,-43.327,-45.766,-45.549,-45.641,-45.242,-45.875,-46.414,-46.227,-44.875,-46.29,-46.459,-46.791,-47.353,-47.22,-48.455,-48.22,-48.031,-48.621,-48.86,-49.649,-49.372,-49.63,-50.696,-50.23,-50.644,-50.823,-50.949,-51.574,-50.917,-50.557,-52.503,-52.126,-52.324,-53.021,-52.555,-53.988],[-22.569,-22.88,-15.792,-22.124,-18.821,-23.346,-21.645,-3.662,-20.756,-17.897,-16.703,-17.165,-14.491,-16.475,-19.144,-6.179,-20.611,-20.97,-17.916,-21.713,-18.902,-22.619,-24.091,-15.832,-23.21,-24.651,-26.807,-26.739,-23.774,-26.246,-29.163,-22.228,-25.772,-29.04,-30.303,-30.832,-30.29,-33.03,-33.946,-32.217,-33.814,-35.011,-37.039,-36.988,-35.854,-39.053,-39.283,-37.679,-38.847,-39.889,-42.156,-41.345,-39.419,-42.753,-42.352,-43.077,-42.24,-43.889,-46.482,-44.909,-43.021,-45.818,-45.404,-45.657,-45.071,-45.756,-46.255,-46.12,-44.701,-46.362,-46.391,-46.931,-47.447,-47.171,-48.508,-48.232,-47.914,-48.701,-48.737,-49.653,-49.308,-49.517,-50.607,-50.094,-50.551,-50.83,-50.883,-51.624,-50.836,-50.338,-52.587,-52.169,-52.357,-53.131,-52.731,-54.171],[-21.062,-22.536,-13.937,-20.819,-18.533,-22.687,-21.954,-3.981,-20.221,-18.349,-16.346,-17.434,-14.263,-16.116,-20.068,-6.035,-20.376,-20.301,-17.784,-21.087,-19.184,-22.292,-24.509,-15.705,-23.748,-25.017,-27.066,-27.462,-24.655,-26.091,-29.637,-22.671,-26.812,-29.208,-30.307,-30.81,-30.411,-32.073,-34.115,-31.144,-33.6,-34.491,-36.308,-36.447,-35.811,-37.786,-39.015,-36.938,-38.487,-39.26,-41.354,-40.768,-39.731,-42.16,-42.336,-42.278,-42.394,-43.572,-45.758,-44.711,-43.732,-45.527,-45.758,-45.554,-45.383,-45.986,-46.706,-46.5,-45.434,-46.419,-46.67,-46.733,-47.224,-47.295,-48.291,-48.169,-48.172,-48.478,-48.896,-49.564,-49.352,-49.687,-50.62,-50.369,-50.8,-50.872,-51.098,-51.67,-51.099,-50.953,-52.474,-52.206,-52.446,-52.997,-52.519,-53.835]],"distant":[[-20.073,-16.399,-12.383,-17.754,-11.219,-19.954,-20.491,-7.831,-15.747,-20.015,-11.269,-13.425,-8.316,-19.409,-20.133,-10.271,-15.073,-17.3,-15.6,-17.81,-13.304,-22.338,-24.038,-18.054,-20.612,-22.728,-23.784,-23.898,-19.753,-26.371,-28.229,-29.57,-28.442,-30.51,-31.408,-31.815,-31.775,-32.141,-32.914,-32.831,-32.492,-33.582,-34.962,-36.211,-33.303,-35.746,-34.82,-36.352,-37.285,-38.102,-38.946,-39.461,-38.255,-40.473,-41.229,-41.159,-42.527,-43.1,-43.949,-43.371,-44.931,-45.569,-45.536,-46.481,-47.827,-48.079,-48.291,-48.537,-48.546,-48.184,-48.524,-50.142,-51.108,-51.201,-50.255,-50.481,-51.754,-51.763,-51.705,-52.238,-52.442,-52.394,-52.331,-52.312,-51.803,-51.529,-51.837,-51.714,-52.453,-52.065,-52.818,-52.006,-51.759,-52.701,-52.526,-52.13],[-20.015,-16.459,-12.244,-17.713,-11.238,-19.87,-20.487,-7.819,-15.744,-19.965,-11.248,-13.42,-8.269,-19.377,-20.171,-10.29,-15.078,-17.397,-15.707,-17.814,-13.402,-22.427,-24.095,-18.151,-20.723,-22.803,-23.914,-24.016,-19.915,-26.496,-28.35,-29.605,-28.46,-30.512,-31.471,-31.874,-31.81,-32.245,-33.027,-32.956,-32.602,-33.705,-35.079,-36.322,-33.428,-35.884,-34.97,-36.488,-37.435,-38.253,-39.089,-39.611,-38.416,-40.616,-41.37,-41.313,-42.676,-43.243,-44.093,-43.541,-45.077,-45.71,-45.683,-46.622,-47.946,-48.197,-48.411,-48.659,-48.656,-48.341,-48.682,-50.246,-51.193,-51.256,-50.328,-50.537,-51.789,-51.778,-51.737,-52.241,-52.438,-52.399,-52.332,-52.302,-51.8,-51.538,-51.825,-51.694,-52.412,-52.051,-52.809,-52.037,-51.802,-52.736,-52.567,-52.206],[-20.36,-15.961,-13.159,-17.942,-10.982,-20.596,-20.479,-7.932,-15.746,-20.388,-11.418,-13.421,-8.715,-19.628,-19.821,-10.223,-15.089,-16.747,-14.942,-17.819,-12.751,-21.817,-23.679,-17.543,-19.965,-22.326,-23.021,-23.162,-18.824,-25.644,-27.495,-29.361,-28.356,-30.553,-31.053,-31.471,-31.58,-31.532,-32.222,-32.096,-31.824,-32.84,-34.255,-35.531,-32.582,-34.93,-33.95,-35.569,-36.395,-37.204,-38.099,-38.571,-37.314,-39.626,-40.399,-40.249,-41.649,-42.254,-43.092,-42.362,-44.063,-44.721,-44.666,-45.636,-47.113,-47.362,-47.553,-47.775,-47.856,-47.182,-47.511,-49.466,-50.523,-50.808,-49.739,-50.102,-51.474,-51.633,-51.467,-52.189,-52.439,-52.353,-52.363,-52.431,-51.901,-51.536,-52.023,-51.99,-52.878,-52.278,-52.986,-51.902,-51.562,-52.57,-52.339,-51.707],[-20.351,-16.071,-13.112,-17.961,-11.064,-20.476,-20.501,-7.905,-15.762,-20.313,-11.384,-13.44,-8.606,-19.589,-19.928,-10.19,-15.064,-16.826,-15.067,-17.814,-12.837,-21.907,-23.749,-17.592,-20.065,-22.376,-23.158,-23.304,-18.975,-25.765,-27.628,-29.404,-28.369,-30.535,-31.108,-31.517,-31.608,-31.631,-32.357,-32.216,-31.951,-32.98,-34.388,-35.66,-32.698,-35.078,-34.098,-35.696,-36.561,-37.373,-38.251,-38.732,-37.48,-39.779,-40.542,-40.413,-41.803,-42.406,-43.255,-42.555,-44.229,-44.884,-44.827,-45.795,-47.244,-47.495,-47.692,-47.926,-47.991,-47.417,-47.761,-49.609,-50.66,-50.902,-49.869,-50.197,-51.568,-51.69,-51.545,-52.238,-52.479,-52.386,-52.362,-52.405,-51.87,-51.524,-51.958,-51.892,-52.752,-52.203,-52.921,-51.887,-51.575,-52.555,-52.344,-51.766],[-20.328,-16.184,-13.029,-17.963,-11.146,-20.339,-20.519,-7.877,-15.773,-20.23,-11.349,-13.455,-8.498,-19.544,-20.028,-10.168,-15.044,-16.922,-15.205,-17.807,-12.936,-22.008,-23.825,-17.661,-20.184,-22.437,-23.312,-23.463,-19.153,-25.904,-27.779,-29.448,-28.381,-30.515,-31.172,-31.575,-31.641,-31.745,-32.505,-32.356,-32.091,-33.136,-34.535,-35.803,-32.833,-35.245,-34.268,-35.843,-36.747,-37.561,-38.423,-38.914,-37.669,-39.952,-40.707,-40.598,-41.979,-42.579,-43.435,-42.77,-44.412,-45.064,-45.007,-45.971,-47.39,-47.643,-47.847,-48.092,-48.14,-47.664,-48.02,-49.764,-50.805,-51.002,-50.003,-50.294,-51.657,-51.74,-51.618,-52.274,-52.505,-52.41,-52.354,-52.374,-51.836,-51.51,-51.892,-51.795,-52.626,-52.128,-52.857,-51.879,-51.594,-52.55,-52.357,-51.837],[-20.37,-15.66,-13.257,-17.879,-10.727,-20.989,-20.411,-8.018,-15.706,-20.653,-11.523,-13.369,-9.066,-19.746,-19.549,-10.33,-15.16,-16.513,-14.585,-17.841,-12.511,-21.564,-23.476,-17.405,-19.674,-22.192,-22.645,-22.756,-18.391,-25.307,-27.116,-29.241,-28.321,-30.623,-30.902,-31.33,-31.502,-31.253,-31.849,-31.751,-31.472,-32.448,-33.886,-35.169,-32.253,-34.522,-33.531,-35.211,-35.935,-36.737,-37.67,-38.119,-36.848,-39.198,-39.991,-39.794,-41.216,-41.829,-42.645,-41.842,-43.616,-44.28,-44.224,-45.201,-46.751,-46.995,-47.166,-47.358,-47.479,-46.586,-46.903,-49.067,-50.143,-50.529,-49.373,-49.83,-51.205,-51.469,-51.242,-52.058,-52.333,-52.263,-52.384,-52.526,-52.031,-51.597,-52.256,-52.342,-53.318,-52.553,-53.222,-51.972,-51.548,-52.627,-52.347,-51.546],[-20.423,-15.841,-13.35,-18.008,-10.864,-20.879,-20.468,-7.973,-15.76,-20.565,-11.471,-13.424,-8.87,-19.707,-19.771,-10.209,-15.093,-16.575,-14.75,-17.851,-12.604,-21.678,-23.574,-17.396,-19.768,-22.225,-22.821,-22.942,-18.55,-25.443,-27.279,-29.318,-28.346,-30.611,-30.966,-31.356,-31.532,-31.359,-32.033,-31.876,-31.646,-32.634,-34.061,-35.337,-32.375,-34.71,-33.699,-35.344,-36.148,-36.953,-37.856,-38.317,-37.044,-39.389,-40.159,-39.997,-41.4,-42.019,-42.862,-42.1,-43.841,-44.499,-44.432,-45.41,-46.918,-47.163,-47.344,-47.564,-47.661,-46.961,-47.32,-49.273,-50.354,-50.678,-49.59,-50.0,-51.397,-51.615,-51.414,-52.213,-52.473,-52.373,-52.423,-52.52,-52.01,-51.6,-52.154,-52.164,-53.108,-52.431,-53.114,-51.917,-51.549,-52.555,-52.322,-51.601],[-20.274,-15.383,-13.091,-17.661,-10.48,-21.196,-20.316,-8.1,-15.622,-20.869,-11.614,-13.279,-9.449,-19.813,-19.24,-10.554,-15.271,-16.389,-14.287,-17.821,-12.343,-21.361,-23.3,-17.419,-19.494,-22.137,-22.358,-22.424,-18.095,-25.075,-26.829,-29.115,-28.265,-30.652,-30.806,-31.269,-31.449,-31.069,-31.546,-31.519,-31.178,-32.13,-33.592,-34.882,-32.028,-34.21,-33.225,-34.974,-35.57,-36.368,-37.338,-37.765,-36.496,-38.868,-39.682,-39.441,-40.896,-41.499,-42.286,-41.428,-43.262,-43.925,-43.876,-44.849,-46.47,-46.713,-46.862,-47.015,-47.174,-46.063,-46.371,-48.723,-49.8,-50.261,-49.011,-49.536,-50.882,-51.22,-50.943,-51.805,-52.097,-52.068,-52.317,-52.547,-52.112,-51.608,-52.467,-52.719,-53.756,-52.824,-53.459,-52.093,-51.569,-52.762,-52.419,-51.459],[-20.091,-15.004,-12.723,-17.239,-10.23,-21.174,-20.178,-8.164,-15.448,-20.949,-11.71,-13.084,-9.968,-19.852,-18.708,-11.022,-15.51,-16.388,-13.982,-17.714,-12.178,-21.133,-23.115,-17.633,-19.426,-22.125,-22.041,-22.095,-17.895,-24.875,-26.584,-28.945,-28.184,-30.601,-30.741,-31.321,-31.425,-30.94,-31.213,-31.372,-30.83,-31.793,-33.278,-34.59,-31.865,-33.889,-32.965,-34.81,-35.195,-35.994,-37.026,-37.429,-36.187,-38.548,-39.406,-39.088,-40.614,-41.168,-41.889,-40.969,-42.857,-43.52,-43.494,-44.45,-46.169,-46.423,-46.543,-46.627,-46.836,-45.342,-45.602,-48.32,-49.375,-49.956,-48.552,-49.136,-50.413,-50.807,-50.481,-51.334,-51.64,-51.693,-52.059,-52.391,-51.971,-51.43,-52.524,-53.01,-54.093,-52.973,-53.599,-52.201,-51.536,-52.988,-52.505,-51.388],[-20.035,-14.874,-12.586,-17.045,-10.118,-21.164,-20.135,-8.152,-15.396,-21.102,-11.753,-12.963,-10.259,-19.905,-18.59,-11.322,-15.648,-16.384,-13.799,-17.63,-12.052,-20.992,-23.031,-17.785,-19.411,-22.119,-21.902,-21.907,-17.785,-24.797,-26.492,-28.917,-28.144,-30.602,-30.81,-31.343,-31.461,-30.909,-31.079,-31.295,-30.641,-31.625,-33.132,-34.453,-31.776,-33.749,-32.796,-34.735,-35.002,-35.804,-36.847,-37.227,-36.011,-38.393,-39.231,-38.89,-40.483,-40.988,-41.701,-40.784,-42.715,-43.346,-43.287,-44.228,-46.021,-46.299,-46.374,-46.442,-46.668,-45.154,-45.543,-48.124,-49.189,-49.798,-48.346,-48.931,-50.183,-50.607,-50.225,-51.142,-51.419,-51.491,-51.879,-52.249,-51.88,-51.274,-52.44,-53.089,-54.179,-53.041,-53.666,-52.238,-51.53,-53.091,-52.596,-51.363],[-20.428,-15.673,-13.394,-17.919,-10.38,-21.48,-20.39,-8.145,-15.794,-21.415,-11.573,-13.408,-9.522,-19.768,-19.892,-10.37,-15.102,-16.116,-14.109,-17.893,-12.253,-21.291,-23.238,-17.199,-19.224,-22.087,-22.406,-22.241,-17.722,-24.98,-26.633,-29.191,-28.182,-30.754,-30.84,-31.031,-31.39,-30.957,-31.579,-31.26,-31.203,-32.079,-33.572,-34.824,-31.791,-34.175,-32.966,-34.761,-35.462,-36.234,-37.124,-37.534,-36.209,-38.727,-39.416,-39.298,-40.705,-41.367,-42.283,-41.49,-43.387,-43.997,-43.865,-44.84,-46.448,-46.67,-46.799,-47.033,-47.15,-46.862,-47.581,-48.742,-49.94,-50.163,-49.06,-49.526,-51.022,-51.352,-51.016,-52.073,-52.323,-52.156,-52.461,-52.755,-52.595,-51.86,-52.727,-53.117,-54.148,-53.257,-53.798,-52.233,-51.751,-52.684,-52.564,-51.454],[-20.731,-16.492,-14.217,-18.995,-10.973,-21.575,-20.639,-8.044,-16.112,-21.018,-11.444,-13.91,-8.534,-19.755,-21.124,-9.513,-14.659,-16.173,-14.951,-18.183,-12.678,-21.877,-23.67,-16.752,-19.409,-22.106,-23.127,-23.135,-18.328,-25.439,-27.256,-29.498,-28.419,-30.871,-30.848,-30.989,-31.415,-31.228,-32.342,-31.647,-32.077,-32.913,-34.312,-35.512,-32.248,-34.956,-33.697,-35.186,-36.457,-37.23,-37.972,-38.486,-37.092,-39.55,-40.216,-40.238,-41.431,-42.242,-43.285,-42.643,-44.326,-44.987,-44.829,-45.881,-47.175,-47.362,-47.604,-48.002,-48.0,-48.687,-49.441,-49.782,-50.972,-50.97,-50.301,-50.64,-52.296,-52.481,-52.367,-53.372,-53.633,-53.26,-53.266,-53.255,-52.944,-52.434,-52.792,-52.587,-53.536,-52.898,-53.462,-51.992,-51.776,-52.283,-52.295,-51.599],[-21.078,-17.191,-14.893,-19.991,-12.16,-21.11,-20.964,-7.668,-16.359,-20.011,-11.266,-14.042,-7.498,-19.873,-21.936,-9.054,-14.523,-16.816,-16.496,-18.24,-13.283,-22.746,-24.461,-16.811,-20.356,-22.247,-24.167,-24.851,-20.005,-26.367,-28.767,-30.032,-28.907,-30.887,-31.261,-31.356,-31.772,-31.991,-33.553,-32.797,-33.198,-34.309,-35.539,-36.745,-33.346,-36.272,-35.214,-36.248,-38.214,-39.085,-39.687,-40.353,-39.046,-41.164,-41.848,-41.945,-43.032,-43.8,-44.889,-44.558,-45.738,-46.439,-46.242,-47.37,-48.29,-48.543,-48.89,-49.477,-49.292,-50.577,-51.05,-51.318,-52.327,-52.378,-52.229,-52.296,-53.85,-53.761,-54.03,-54.742,-54.888,-54.402,-53.528,-52.936,-52.067,-52.023,-51.701,-51.015,-51.953,-51.667,-52.45,-51.384,-51.467,-51.878,-51.88,-51.8],[-21.391,-17.747,-14.944,-19.733,-13.725,-19.482,-21.273,-7.189,-16.306,-18.782,-10.844,-13.544,-6.918,-19.372,-21.369,-9.268,-14.799,-18.287,-18.111,-17.747,-13.941,-23.365,-25.297,-17.871,-22.22,-22.636,-25.422,-26.948,-22.683,-27.722,-30.887,-30.506,-28.998,-30.253,-32.106,-32.254,-32.233,-33.253,-35.049,-34.817,-34.176,-36.065,-37.066,-38.393,-34.853,-37.865,-37.39,-38.069,-40.411,-41.521,-42.005,-42.794,-41.896,-43.321,-43.818,-44.098,-45.401,-45.654,-46.701,-46.938,-47.29,-47.97,-47.925,-48.817,-49.554,-49.932,-50.351,-51.022,-50.593,-51.93,-51.761,-52.643,-53.509,-53.474,-53.383,-52.798,-54.209,-53.471,-53.831,-53.935,-53.9,-53.568,-52.184,-51.605,-50.408,-50.521,-50.02,-49.415,-50.279,-50.228,-51.179,-50.723,-50.907,-51.665,-51.556,-52.082],[-20.865,-18.271,-13.608,-17.79,-14.239,-17.588,-21.111,-7.1,-15.547,-18.003,-10.438,-13.205,-7.02,-18.302,-20.035,-10.129,-15.195,-20.009,-18.175,-16.932,-14.359,-23.21,-25.301,-19.895,-23.91,-23.212,-26.292,-27.565,-25.162,-29.099,-31.891,-30.031,-28.08,-29.218,-32.345,-33.541,-32.122,-34.575,-35.97,-36.699,-34.584,-36.837,-37.816,-39.325,-36.101,-39.004,-39.261,-40.226,-41.746,-42.829,-43.175,-44.014,-43.566,-44.461,-44.799,-45.369,-47.135,-46.767,-47.519,-48.27,-48.206,-48.844,-49.294,-49.5,-50.291,-50.79,-51.245,-51.483,-51.103,-51.841,-51.284,-52.762,-53.697,-52.796,-52.017,-50.978,-52.75,-51.654,-51.807,-51.48,-51.549,-51.465,-50.627,-50.653,-49.593,-49.634,-49.431,-49.108,-49.615,-49.599,-50.529,-50.66,-50.66,-51.985,-51.723,-52.702],[-19.419,-18.169,-11.157,-15.982,-13.18,-16.836,-20.475,-7.494,-14.785,-18.255,-10.55,-13.207,-7.304,-18.116,-19.681,-11.404,-15.324,-20.779,-17.626,-16.68,-14.945,-23.429,-24.754,-21.816,-24.282,-24.001,-26.755,-27.144,-26.975,-30.228,-31.739,-29.13,-27.501,-28.984,-32.037,-34.317,-31.947,-35.322,-35.796,-37.405,-34.678,-36.457,-37.747,-39.265,-37.049,-39.72,-40.071,-41.509,-41.837,-42.705,-43.044,-43.927,-43.636,-44.535,-45.303,-45.856,-47.667,-47.17,-47.587,-48.588,-48.723,-49.319,-49.876,-49.994,-50.673,-51.271,-51.726,-51.523,-51.322,-51.498,-51.256,-52.704,-53.406,-52.12,-50.96,-50.075,-51.542,-50.601,-50.659,-50.282,-50.558,-50.617,-50.443,-50.7,-50.168,-50.064,-50.261,-50.141,-50.23,-50.235,-51.16,-51.688,-51.48,-52.98,-52.736,-53.732],[-18.122,-17.551,-9.198,-15.217,-11.856,-17.035,-19.987,-7.97,-14.9,-19.227,-11.101,-13.212,-7.462,-18.552,-20.302,-12.413,-15.066,-20.75,-17.71,-17.15,-16.317,-24.53,-24.636,-23.251,-24.378,-25.249,-27.753,-27.023,-28.363,-31.009,-31.585,-28.987,-27.941,-29.599,-32.232,-33.94,-32.094,-35.454,-35.308,-37.426,-34.918,-36.286,-37.737,-39.1,-37.889,-40.346,-40.445,-41.865,-41.635,-42.513,-43.025,-44.042,-43.964,-44.713,-46.001,-46.461,-47.829,-47.476,-47.806,-48.985,-49.276,-49.7,-50.1,-50.834,-51.37,-51.838,-52.244,-52.033,-51.791,-51.614,-51.898,-52.909,-53.093,-52.127,-51.02,-50.622,-50.942,-50.358,-50.538,-50.406,-50.727,-50.915,-51.389,-51.316,-51.594,-51.338,-51.837,-51.929,-51.867,-51.963,-52.878,-53.608,-53.25,-54.403,-54.338,-54.723],[-17.483,-17.08,-8.075,-15.413,-10.705,-17.887,-19.693,-8.139,-15.69,-20.044,-11.505,-13.239,-7.762,-18.792,-20.871,-12.718,-14.966,-20.616,-18.27,-17.771,-18.078,-25.808,-25.098,-24.676,-24.874,-26.737,-29.273,-26.999,-29.104,-31.635,-31.815,-29.77,-28.81,-30.713,-33.052,-33.114,-32.437,-35.672,-35.302,-37.371,-35.367,-36.678,-37.936,-38.841,-38.507,-40.713,-40.964,-42.087,-41.701,-42.637,-43.454,-44.589,-44.755,-45.188,-46.737,-47.105,-47.998,-48.073,-48.33,-49.561,-50.017,-49.968,-50.613,-51.767,-52.255,-52.407,-52.675,-52.549,-52.317,-51.693,-52.321,-52.846,-52.536,-51.996,-51.189,-51.334,-50.731,-50.663,-51.053,-51.131,-51.175,-51.668,-52.645,-52.18,-53.145,-52.925,-53.59,-53.974,-53.881,-54.136,-54.872,-55.565,-55.09,-55.828,-55.901,-55.574],[-17.626,-17.039,-7.603,-16.252,-9.62,-19.608,-19.367,-7.922,-16.332,-20.276,-11.311,-13.165,-8.405,-18.997,-20.864,-12.527,-15.59,-21.019,-19.071,-18.483,-19.187,-26.697,-25.849,-26.199,-25.682,-28.069,-30.646,-26.776,-29.162,-32.614,-32.357,-31.266,-29.629,-32.432,-34.606,-32.963,-33.26,-36.446,-36.158,-37.776,-36.262,-37.818,-38.699,-38.818,-39.278,-41.117,-41.894,-42.723,-42.407,-43.267,-44.637,-45.571,-45.89,-46.174,-47.649,-48.013,-48.671,-49.276,-49.355,-50.531,-51.089,-50.574,-51.773,-52.717,-53.042,-52.892,-52.883,-52.888,-52.628,-51.676,-52.159,-52.36,-51.834,-51.538,-50.92,-51.582,-51.147,-51.723,-52.109,-52.247,-51.908,-52.848,-54.015,-53.487,-54.536,-54.596,-55.138,-55.702,-55.345,-56.041,-56.569,-56.912,-56.587,-57.09,-57.261,-56.697],[-18.328,-17.188,-7.734,-16.858,-8.97,-21.512,-19.446,-7.552,-15.975,-20.102,-10.785,-12.946,-9.104,-19.284,-20.413,-11.974,-16.649,-22.159,-19.832,-19.744,-19.416,-27.142,-26.449,-27.378,-26.069,-28.756,-31.096,-26.515,-28.686,-33.595,-32.496,-32.763,-30.513,-34.462,-36.517,-34.001,-34.475,-37.483,-37.41,-38.654,-37.717,-39.223,-40.189,-39.652,-40.511,-41.974,-43.219,-43.86,-43.858,-44.196,-46.322,-46.574,-46.885,-47.48,-48.622,-49.42,-49.946,-50.67,-50.858,-51.896,-52.074,-51.916,-53.127,-53.497,-53.338,-52.845,-52.509,-52.893,-52.084,-51.444,-51.441,-51.705,-51.409,-51.108,-50.516,-51.677,-52.043,-53.132,-53.444,-53.595,-53.115,-54.305,-55.272,-55.013,-55.383,-55.62,-55.665,-56.299,-55.864,-56.984,-57.477,-57.415,-57.576,-57.945,-58.126,-57.796],[-18.594,-16.778,-7.973,-16.662,-8.849,-21.784,-20.264,-7.495,-15.143,-19.668,-10.562,-12.581,-9.316,-19.245,-19.722,-11.677,-17.396,-23.115,-20.358,-21.403,-20.177,-27.421,-26.689,-27.602,-25.924,-28.811,-30.678,-26.715,-28.419,-33.608,-32.319,-33.509,-31.656,-35.73,-37.404,-35.298,-35.16,-37.947,-37.894,-39.175,-39.057,-39.693,-41.52,-40.881,-41.688,-43.025,-44.392,-44.938,-45.456,-44.862,-47.33,-47.049,-47.295,-48.35,-49.26,-50.525,-51.025,-51.399,-52.051,-52.808,-52.409,-53.226,-53.689,-53.587,-52.914,-52.071,-51.647,-51.948,-50.78,-50.861,-50.722,-51.459,-51.453,-51.036,-50.611,-52.109,-52.893,-53.996,-54.451,-54.604,-54.399,-55.271,-55.845,-55.808,-55.503,-55.674,-55.435,-56.185,-56.091,-57.025,-57.506,-57.3,-57.872,-58.012,-57.84,-57.72],[-17.576,-15.855,-7.917,-16.249,-8.706,-20.986,-20.812,-8.021,-14.882,-19.015,-10.786,-12.085,-9.091,-18.824,-19.027,-11.902,-17.848,-23.459,-20.775,-22.634,-21.938,-27.804,-26.564,-27.431,-26.057,-28.788,-30.3,-27.677,-28.645,-33.213,-32.589,-33.807,-32.921,-36.122,-37.401,-35.773,-35.283,-38.01,-38.106,-39.164,-39.751,-39.711,-42.047,-41.681,-42.577,-43.979,-45.182,-45.831,-46.6,-45.477,-47.634,-47.531,-47.65,-48.894,-49.818,-50.867,-51.661,-51.808,-52.65,-53.008,-52.492,-53.556,-53.518,-52.964,-52.214,-51.242,-50.714,-50.486,-49.776,-50.553,-50.418,-51.654,-51.752,-51.665,-51.543,-53.094,-53.74,-54.486,-55.054,-55.148,-55.292,-55.565,-56.109,-56.167,-55.547,-55.913,-55.68,-56.306,-56.537,-56.922,-57.297,-57.156,-57.638,-57.383,-56.802,-56.809],[-15.999,-15.142,-8.056,-16.11,-8.699,-20.551,-20.096,-8.647,-15.354,-18.34,-10.958,-11.681,-8.754,-18.475,-18.509,-11.818,-17.969,-23.84,-20.986,-23.109,-23.432,-28.153,-26.355,-27.45,-26.545,-28.76,-30.241,-29.094,-28.875,-33.082,-33.261,-34.113,-34.043,-36.257,-37.457,-35.84,-35.425,-38.196,-38.645,-39.129,-40.142,-40.09,-42.319,-42.287,-43.389,-44.79,-45.836,-46.635,-47.333,-46.232,-47.916,-48.39,-48.127,-49.511,-50.612,-50.89,-52.085,-52.159,-52.973,-53.027,-52.43,-53.081,-52.915,-51.993,-51.447,-50.436,-49.847,-49.448,-49.343,-50.644,-50.414,-51.967,-52.145,-52.757,-52.705,-54.167,-54.566,-54.793,-55.378,-55.295,-55.699,-55.578,-56.326,-56.591,-55.782,-56.498,-56.319,-56.497,-56.854,-56.547,-56.752,-56.967,-56.871,-56.355,-55.774,-55.91],[-14.849,-15.029,-8.914,-16.652,-9.453,-20.894,-18.885,-8.502,-16.071,-17.529,-10.663,-11.43,-8.44,-18.501,-18.018,-10.849,-17.577,-24.376,-20.926,-23.31,-23.207,-28.154,-26.315,-27.588,-26.892,-28.666,-30.017,-30.085,-28.528,-33.202,-33.729,-34.434,-34.942,-36.3,-37.614,-36.012,-35.658,-38.545,-39.186,-39.2,-40.463,-40.682,-42.729,-43.064,-43.93,-45.296,-46.538,-47.06,-47.675,-46.844,-48.251,-49.282,-48.482,-50.019,-51.475,-50.811,-52.15,-52.143,-52.664,-52.841,-51.786,-51.889,-51.687,-50.797,-50.473,-49.521,-49.111,-48.86,-48.953,-50.652,-50.72,-52.305,-52.492,-53.483,-53.275,-54.654,-54.952,-54.83,-55.347,-55.099,-55.753,-55.392,-56.291,-56.849,-55.956,-56.586,-56.405,-56.055,-56.243,-55.442,-55.623,-56.421,-55.645,-55.136,-55.04,-55.404],[-14.768,-15.499,-10.866,-18.509,-11.422,-22.13,-18.083,-7.377,-16.349,-16.113,-10.018,-11.469,-8.256,-18.971,-17.452,-9.653,-17.067,-24.88,-20.965,-23.764,-21.657,-27.741,-26.532,-27.785,-26.684,-28.447,-29.093,-29.804,-27.225,-33.427,-33.304,-34.341,-35.339,-35.933,-37.724,-36.357,-36.017,-38.936,-39.565,-39.518,-40.664,-40.939,-43.137,-43.861,-43.842,-45.036,-47.178,-46.943,-47.275,-46.958,-48.207,-49.882,-48.948,-50.33,-52.233,-51.013,-52.178,-51.663,-51.222,-52.202,-50.581,-49.803,-49.503,-49.665,-49.723,-48.893,-48.591,-48.501,-48.689,-50.72,-51.591,-52.717,-52.844,-53.6,-53.349,-54.247,-54.73,-54.651,-55.008,-54.628,-55.592,-55.005,-56.006,-56.58,-55.716,-55.539,-55.137,-54.629,-54.71,-53.969,-54.382,-55.492,-54.328,-54.104,-54.902,-55.743],[-15.987,-15.916,-13.798,-22.052,-14.317,-24.064,-18.024,-6.196,-15.8,-14.335,-9.345,-11.883,-8.191,-19.626,-17.096,-9.284,-17.272,-25.276,-21.59,-24.735,-20.308,-27.228,-27.023,-28.224,-26.14,-28.175,-27.845,-28.597,-25.635,-33.46,-32.119,-33.781,-35.056,-35.283,-37.908,-36.784,-36.649,-39.038,-39.743,-39.949,-40.673,-40.231,-42.903,-44.062,-43.308,-43.523,-47.193,-46.506,-46.123,-46.313,-47.171,-50.09,-49.825,-50.512,-52.686,-51.748,-52.681,-50.867,-49.589,-51.416,-49.228,-47.506,-47.113,-49.249,-49.895,-48.876,-48.434,-48.763,-49.444,-51.454,-52.721,-53.27,-53.47,-53.623,-53.39,-53.318,-54.262,-54.421,-54.629,-54.089,-55.506,-54.543,-55.729,-55.715,-55.039,-54.089,-53.034,-52.987,-53.259,-52.905,-53.722,-54.508,-53.612,-53.863,-55.782,-57.399],[-15.642,-15.482,-10.217,-18.184,-10.795,-22.037,-18.685,-7.312,-15.754,-15.822,-10.03,-11.718,-8.363,-19.011,-17.726,-10.287,-17.547,-24.615,-21.227,-23.791,-21.587,-27.702,-26.664,-27.878,-26.482,-28.499,-28.965,-29.009,-27.004,-33.286,-32.761,-34.079,-34.671,-35.822,-37.73,-36.312,-36.023,-38.635,-39.225,-39.447,-40.408,-40.218,-42.612,-43.184,-43.443,-44.032,-46.553,-46.595,-46.706,-46.382,-47.491,-49.307,-48.889,-50.036,-51.657,-51.206,-52.304,-51.431,-51.079,-52.156,-50.422,-49.382,-48.992,-50.311,-50.46,-49.4,-49.034,-49.107,-49.356,-51.038,-51.429,-52.643,-52.804,-53.279,-53.087,-53.813,-54.472,-54.615,-54.963,-54.651,-55.58,-55.002,-56.044,-56.124,-55.455,-55.277,-54.35,-54.406,-54.658,-54.281,-54.875,-55.472,-54.872,-54.842,-55.635,-56.42],[-15.769,-15.553,-11.101,-19.087,-11.625,-22.617,-18.469,-6.945,-15.727,-15.088,-9.759,-11.78,-8.241,-19.202,-17.531,-10.057,-17.591,-24.883,-21.438,-24.197,-21.191,-27.577,-26.81,-28.066,-26.434,-28.438,-28.552,-28.879,-26.452,-33.295,-32.475,-34.027,-34.881,-35.68,-37.839,-36.515,-36.27,-38.716,-39.376,-39.532,-40.481,-40.06,-42.603,-43.426,-43.432,-43.58,-46.74,-46.558,-46.429,-46.344,-47.161,-49.578,-49.22,-50.209,-51.994,-51.412,-52.477,-51.16,-50.572,-51.88,-49.835,-48.476,-48.016,-50.016,-50.36,-49.201,-48.888,-49.13,-49.527,-51.281,-51.783,-52.913,-53.079,-53.469,-53.276,-53.672,-54.446,-54.588,-54.851,-54.493,-55.6,-54.832,-55.99,-55.86,-55.28,-54.884,-53.632,-53.8,-54.084,-53.784,-54.505,-55.022,-54.493,-54.626,-55.784,-56.799],[-15.904,-15.555,-9.779,-17.851,-10.426,-21.854,-18.908,-7.406,-15.648,-16.126,-10.129,-11.804,-8.442,-18.995,-17.936,-10.533,-17.566,-24.352,-21.126,-23.402,-21.519,-27.69,-26.642,-27.802,-26.426,-28.535,-29.145,-28.73,-27.193,-33.266,-32.711,-33.969,-34.186,-35.756,-37.617,-36.139,-35.862,-38.488,-38.989,-39.361,-40.163,-40.059,-42.377,-42.782,-43.17,-43.847,-46.166,-46.347,-46.498,-46.153,-47.373,-48.927,-48.638,-49.753,-51.245,-51.075,-52.112,-51.4,-51.203,-52.208,-50.609,-49.691,-49.325,-50.643,-50.732,-49.662,-49.299,-49.368,-49.521,-51.032,-51.305,-52.507,-52.641,-53.003,-52.796,-53.631,-54.267,-54.52,-54.882,-54.642,-55.413,-55.006,-56.009,-56.024,-55.437,-55.343,-54.442,-54.599,-54.841,-54.575,-55.164,-55.651,-55.193,-55.176,-55.84,-56.516],[-15.888,-15.552,-10.208,-18.263,-10.814,-22.129,-18.761,-7.229,-15.655,-15.684,-9.984,-11.803,-8.364,-19.075,-17.792,-10.38,-17.603,-24.543,-21.257,-23.7,-21.4,-27.653,-26.713,-27.911,-26.431,-28.509,-28.915,-28.763,-26.892,-33.271,-32.601,-33.995,-34.428,-35.739,-37.709,-36.284,-36.017,-38.562,-39.119,-39.416,-40.276,-40.028,-42.444,-42.994,-43.263,-43.695,-46.361,-46.417,-46.458,-46.211,-47.258,-49.145,-48.848,-49.914,-51.497,-51.211,-52.255,-51.299,-50.965,-52.084,-50.286,-49.172,-48.75,-50.43,-50.622,-49.498,-49.165,-49.313,-49.554,-51.14,-51.468,-52.656,-52.8,-53.161,-52.964,-53.638,-54.331,-54.553,-54.873,-54.597,-55.489,-54.944,-56.013,-55.941,-55.371,-55.168,-54.078,-54.272,-54.536,-54.277,-54.929,-55.394,-54.948,-55.005,-55.855,-56.652],[-16.221,-15.713,-9.167,-17.309,-9.956,-21.381,-19.203,-7.598,-15.586,-16.871,-10.346,-11.939,-8.545,-18.94,-18.318,-10.885,-17.398,-23.8,-20.794,-22.476,-21.206,-27.573,-26.519,-27.449,-26.297,-28.466,-29.461,-28.365,-27.605,-33.179,-32.708,-33.614,-33.26,-35.35,-37.186,-35.725,-35.45,-38.185,-38.533,-39.163,-39.565,-39.795,-41.868,-42.09,-42.531,-43.59,-45.364,-45.791,-46.004,-45.722,-47.134,-48.237,-48.07,-49.129,-50.475,-50.611,-51.586,-51.24,-51.276,-52.171,-50.979,-50.376,-50.151,-51.117,-51.092,-50.106,-49.734,-49.746,-49.739,-50.985,-51.158,-52.304,-52.389,-52.613,-52.344,-53.314,-53.844,-54.197,-54.589,-54.446,-54.938,-54.854,-55.762,-55.766,-55.327,-55.356,-54.734,-54.941,-55.124,-54.99,-55.536,-55.959,-55.604,-55.589,-56.001,-56.48]]},"fill":{"onboard":[[-30.758,-20.256,-14.815,-15.594,-15.819,-19.496],[-30.138,-19.761,-14.73,-15.642,-16.089,-20.218],[-33.059,-21.928,-14.881,-15.261,-14.954,-18.128],[-32.185,-21.177,-14.491,-15.201,-15.096,-19.244],[-33.619,-22.242,-14.481,-14.709,-14.374,-18.314],[-33.512,-22.39,-14.115,-14.101,-13.931,-18.914],[-32.347,-20.788,-12.836,-13.665,-13.225,-20.33],[-31.3,-19.585,-11.673,-12.914,-11.974,-20.327],[-31.981,-19.531,-12.261,-14.192,-13.733,-22.611],[-32.945,-20.702,-14.112,-16.539,-17.31,-26.485],[-31.624,-20.973,-15.904,-18.31,-20.992,-29.473],[-28.65,-19.375,-16.397,-17.933,-23.383,-29.422],[-25.872,-16.878,-15.913,-16.313,-24.502,-26.81],[-23.999,-14.376,-15.739,-15.097,-25.081,-23.639],[-23.477,-13.141,-16.247,-15.509,-25.454,-21.915],[-23.811,-13.684,-16.259,-17.718,-25.593,-21.866],[-23.505,-14.713,-15.078,-20.525,-24.838,-21.925],[-22.442,-15.137,-14.017,-22.97,-23.736,-21.68],[-21.652,-15.377,-14.172,-25.065,-23.388,-22.057],[-21.968,-16.058,-16.075,-27.441,-24.228,-23.716],[-23.291,-15.632,-19.305,-29.701,-25.513,-25.738],[-23.96,-13.949,-21.133,-30.113,-25.906,-25.772],[-24.312,-13.546,-21.872,-29.801,-26.264,-24.738],[-25.91,-14.562,-23.438,-29.822,-27.377,-23.345],[-28.382,-15.868,-25.528,-30.071,-28.809,-22.44],[-30.523,-17.584,-27.767,-30.865,-30.319,-23.244],[-30.686,-18.566,-28.229,-31.441,-30.823,-24.601],[-29.53,-17.882,-27.014,-31.3,-30.171,-24.744],[-27.547,-16.728,-24.16,-30.57,-28.765,-24.524],[-28.41,-17.358,-25.35,-31.003,-29.443,-24.836],[-25.924,-16.099,-21.062,-28.428,-27.435,-24.234]],"paddock":[[-15.522,-5.651,-5.274,-6.275,-10.294,-16.361],[-15.44,-5.006,-5.214,-6.191,-10.148,-16.313],[-15.368,-4.599,-5.116,-6.028,-9.988,-16.254],[-15.473,-5.91,-5.135,-6.095,-9.966,-16.409],[-15.607,-7.624,-5.327,-6.415,-10.508,-16.495],[-15.912,-9.439,-5.664,-6.771,-11.558,-16.446],[-16.838,-10.988,-6.298,-7.097,-12.809,-16.351],[-19.658,-12.356,-7.384,-7.48,-13.267,-16.065],[-25.032,-13.316,-7.39,-7.836,-12.954,-15.569],[-27.646,-13.248,-6.252,-7.95,-13.318,-15.643],[-27.064,-12.56,-5.525,-7.575,-14.266,-16.253],[-26.522,-11.585,-5.735,-7.214,-14.941,-16.914],[-26.58,-10.673,-6.557,-7.904,-14.416,-17.114],[-26.706,-10.082,-7.096,-9.542,-13.274,-17.035],[-26.425,-9.71,-7.167,-11.31,-12.413,-17.152],[-25.805,-9.371,-7.661,-12.806,-12.4,-17.563],[-25.263,-8.741,-8.983,-13.761,-13.013,-17.798],[-25.254,-8.062,-10.354,-13.795,-13.434,-17.902],[-25.991,-8.097,-11.222,-13.408,-12.943,-18.333],[-27.253,-9.257,-11.993,-13.576,-11.386,-18.858],[-28.7,-11.339,-13.035,-14.707,-9.935,-19.23],[-27.214,-9.671,-11.884,-14.182,-11.221,-18.826],[-27.821,-10.301,-12.477,-14.555,-10.844,-19.107],[-27.031,-9.617,-11.43,-14.084,-11.409,-18.72],[-26.532,-9.324,-10.274,-13.321,-11.886,-18.329],[-26.658,-9.403,-10.54,-13.523,-11.763,-18.429],[-26.795,-9.501,-10.801,-13.708,-11.643,-18.53],[-26.122,-9.355,-9.561,-12.532,-12.117,-18.08],[-26.241,-9.368,-9.725,-12.708,-12.05,-18.144],[-26.348,-9.388,-9.887,-12.872,-11.983,-18.208],[-24.759,-9.422,-8.742,-11.388,-12.33,-17.747]],"outside":[[-18.905,-21.955,-17.005,-9.149,-24.963,-28.406],[-18.924,-21.84,-16.982,-9.228,-24.915,-28.314],[-18.956,-21.729,-16.969,-9.326,-24.872,-28.228],[-19.003,-21.621,-16.964,-9.443,-24.834,-28.148],[-18.775,-22.563,-17.127,-8.705,-25.242,-28.906],[-18.689,-22.337,-17.015,-8.681,-25.121,-28.686],[-19.297,-23.557,-17.651,-9.028,-25.802,-29.96],[-18.846,-23.139,-17.371,-8.541,-25.625,-29.514],[-20.095,-24.208,-17.978,-9.929,-26.175,-30.783],[-21.778,-24.872,-18.229,-12.608,-26.337,-31.509],[-22.412,-24.671,-17.64,-14.841,-26.468,-31.504],[-19.244,-23.32,-16.744,-8.965,-26.766,-30.769],[-16.803,-22.652,-17.186,-5.525,-26.491,-29.668],[-15.937,-22.188,-17.415,-4.763,-25.328,-28.18],[-16.775,-21.435,-16.277,-6.68,-24.025,-27.013],[-18.79,-20.819,-15.217,-11.477,-23.369,-26.651],[-20.542,-20.813,-15.458,-17.418,-23.524,-26.857],[-21.623,-20.893,-16.727,-19.896,-24.102,-27.366],[-22.468,-20.291,-18.659,-19.531,-24.872,-28.246],[-23.576,-19.453,-21.216,-19.629,-25.891,-29.148],[-25.2,-19.77,-23.876,-21.412,-27.027,-28.931],[-26.673,-20.521,-25.305,-23.872,-27.725,-27.711],[-27.718,-19.839,-26.134,-24.671,-27.879,-26.502],[-28.914,-18.886,-27.274,-24.6,-27.873,-25.844],[-30.467,-18.732,-28.402,-25.366,-28.232,-25.73],[-31.913,-19.397,-29.063,-26.972,-29.107,-26.077],[-31.589,-19.835,-28.882,-26.563,-29.186,-26.583],[-30.508,-19.757,-28.213,-25.87,-28.837,-26.591],[-28.752,-19.685,-26.63,-24.714,-28.27,-26.676],[-29.375,-19.778,-27.219,-25.137,-28.515,-26.715],[-26.946,-19.747,-24.332,-22.909,-27.602,-26.922]],"distant":[[-17.984,-21.509,-21.343,-20.192,-21.624,-24.095],[-18.069,-21.508,-21.364,-20.291,-21.704,-24.184],[-17.495,-21.492,-21.134,-19.603,-21.014,-23.518],[-17.568,-21.511,-21.224,-19.706,-21.203,-23.654],[-17.655,-21.526,-21.306,-19.823,-21.388,-23.794],[-17.288,-21.441,-20.881,-19.316,-20.521,-23.164],[-17.353,-21.499,-21.084,-19.446,-20.878,-23.395],[-17.179,-21.346,-20.56,-19.098,-20.01,-22.83],[-17.164,-21.16,-20.069,-18.857,-19.268,-22.321],[-17.244,-21.088,-19.876,-18.739,-19.045,-22.164],[-17.096,-21.515,-20.902,-19.111,-20.769,-23.352],[-17.014,-21.861,-22.097,-19.663,-22.786,-24.508],[-17.521,-22.121,-23.301,-20.416,-24.931,-25.384],[-18.798,-22.096,-23.795,-21.159,-25.953,-25.482],[-20.027,-21.444,-22.762,-21.71,-24.521,-24.913],[-20.462,-20.728,-20.985,-22.236,-22.393,-24.84],[-20.608,-20.596,-19.864,-22.9,-21.157,-25.504],[-21.007,-21.106,-19.688,-23.875,-20.711,-26.331],[-21.899,-22.081,-20.343,-25.139,-20.799,-26.86],[-22.91,-22.609,-21.522,-26.035,-21.614,-26.861],[-22.716,-21.766,-22.404,-25.681,-23.025,-26.788],[-21.481,-20.554,-22.479,-24.532,-24.103,-26.82],[-20.373,-20.022,-22.301,-23.449,-24.41,-26.717],[-19.78,-20.166,-22.6,-22.388,-24.598,-26.206],[-19.809,-20.412,-23.706,-20.964,-25.251,-24.841],[-20.159,-20.07,-25.248,-19.526,-26.407,-23.232],[-20.141,-20.179,-23.54,-21.04,-25.209,-24.704],[-20.117,-20.115,-24.03,-20.424,-25.618,-24.142],[-20.335,-20.271,-23.295,-21.404,-24.872,-24.945],[-20.267,-20.21,-23.556,-21.014,-25.143,-24.627],[-20.608,-20.476,-22.782,-22.126,-24.179,-25.425]]},"sub":{"onboard":[-37.688,-37.448,-38.205,-37.857,-37.904,-37.425,-36.607,-35.636,-36.815,-38.408,-38.713,-37.919,-36.729,-35.214,-33.588,-32.563,-31.982,-31.464,-31.032,-30.895,-31.491,-33.123,-34.84,-34.891,-34.055,-34.187,-34.985,-35.069,-34.577,-34.888,-33.835],"paddock":[-15.219,-15.586,-15.793,-14.939,-14.368,-14.173,-14.698,-17.202,-22.938,-27.714,-28.734,-29.91,-31.529,-32.764,-33.681,-34.201,-34.669,-35.499,-36.46,-37.08,-37.508,-36.831,-37.194,-36.529,-35.34,-35.679,-35.955,-32.088,-32.728,-33.305,-26.851],"outside":[-23.346,-23.299,-23.257,-23.221,-23.628,-23.503,-24.184,-24.011,-24.466,-24.425,-24.177,-24.624,-25.326,-24.512,-22.555,-21.466,-21.635,-22.369,-23.269,-24.533,-26.481,-28.337,-29.023,-29.134,-29.551,-30.152,-29.788,-29.462,-28.77,-29.004,-27.571],"distant":[-18.614,-18.762,-17.703,-17.895,-18.103,-17.192,-17.474,-16.757,-16.217,-16.011,-17.051,-18.368,-20.1,-22.01,-22.855,-22.195,-21.581,-21.801,-23.043,-24.731,-24.891,-23.421,-22.056,-21.419,-21.806,-23.305,-22.426,-22.704,-22.584,-22.656,-22.691]},"ovr":[1.738,-5.072,4.501,2.005,-0.551,-0.202,-1.923,1.346,-0.443,5.497,4.568,-1.013,-0.636,1.88,0.202,-2.139,-1.116,-1.517,-1.099,-1.467,-1.212,1.269,-1.912,0.615,-0.902,1.493,-0.443,-0.28,-1.504,-1.617,-0.973,1.512,-0.133,-0.227,-0.678,-1.535,-1.414,-2.503,-0.693,0.932,-1.855,-1.061,-0.251,-0.32,-0.443,-0.595,-0.729,0.44,-0.364,2.194,0.918,0.296,1.597,1.202,0.767,0.549,1.073,0.182,1.267,0.809,0.994,0.353,0.926,1.479,1.096,1.926,2.141,1.432,0.905,0.466,0.674,1.696,0.869,1.061,0.843,0.251,0.646,0.892,1.319,1.682,0.937,1.876,3.074,1.986,2.038,1.227,0.753,1.183,0.898,0.936,0.227,0.933,0.614,0.601,0.771,0.67],"cal":{"onboard":{"noise_db":[-30.7,-12.2,-16.7,-17.1],"jit_db":[1.0,2.0,3.0,4.0],"eq_db":[0.4,0.1,-3.2,-4.5,-4.5],"corr_db":[[-14.93,-11.96,1.87,4.79,7.97,5.67,6.57],[-14.93,-11.96,1.87,4.79,7.97,5.67,6.57],[-14.93,-11.96,1.87,4.79,7.97,5.67,6.57],[-14.93,-11.96,1.87,4.79,7.97,5.67,6.57],[-10.95,-8.63,0.86,2.72,6.22,4.8,5.0],[-6.98,-5.32,-0.15,0.66,4.45,3.93,3.42],[-4.86,-3.87,-0.87,-0.24,3.46,3.43,2.96],[-2.75,-2.44,-1.6,-1.14,2.47,2.92,2.53],[-2.98,-2.87,-0.6,-1.36,2.03,2.97,2.8],[-3.23,-3.31,0.4,-1.58,1.61,3.02,3.09],[-3.96,-3.77,0.52,-1.53,1.57,3.52,3.66],[-4.72,-4.23,0.63,-1.48,1.54,4.01,4.25],[-5.04,-3.74,-0.75,-1.24,1.81,4.26,4.7],[-5.36,-3.24,-2.14,-1.0,2.06,4.51,5.15],[-4.68,-2.57,-2.54,-1.12,2.01,4.15,4.75],[-4.01,-1.91,-2.91,-1.25,1.96,3.79,4.35],[-3.79,-2.38,-2.53,-1.2,1.94,3.82,4.15],[-3.56,-2.84,-2.14,-1.16,1.9,3.84,3.97],[-3.71,-2.82,-2.35,-0.97,1.96,4.03,3.87],[-3.84,-2.8,-2.56,-0.78,2.02,4.21,3.76],[-3.96,-1.58,-2.28,-0.94,2.12,3.87,2.76],[-4.06,-0.35,-2.01,-1.09,2.22,3.56,1.76],[-6.24,0.24,-0.55,-0.44,2.54,3.66,0.79],[-8.42,0.86,0.91,0.19,2.86,3.78,-0.18],[-9.7,0.92,1.41,1.21,3.03,4.1,-0.95],[-10.98,0.99,1.89,2.25,3.19,4.41,-1.74],[-9.38,1.25,1.2,2.16,3.13,4.33,-2.7],[-7.75,1.54,0.51,2.07,3.05,4.27,-3.66],[-7.75,1.54,0.51,2.07,3.05,4.27,-3.66],[-7.75,1.54,0.51,2.07,3.05,4.27,-3.66],[-7.75,1.54,0.51,2.07,3.05,4.27,-3.66]],"fill_db":-18.183366175522767,"skirt":0.15,"whine_db":[-21.0,-19.9,-26.3,-21.9,-32.0,-35.0]},"paddock":{"noise_db":[-21.3,-2.8,-27.7,-17.7],"jit_db":[1.5,2.5,3.5,4.0],"eq_db":[0.5,-13.2,-18.7,-6.8,-6.8],"corr_db":[[-12.58,-8.16,-2.1,3.26,7.23,9.23,3.11],[-12.58,-8.16,-2.1,3.26,7.23,9.23,3.11],[-12.58,-8.16,-2.1,3.26,7.23,9.23,3.11],[-12.58,-8.16,-2.1,3.26,7.23,9.23,3.11],[-14.36,-9.26,-2.16,3.45,8.14,10.01,4.17],[-15.0,-10.35,-2.2,3.62,9.03,10.77,5.26],[-15.0,-13.46,-1.96,4.34,10.74,12.36,8.02],[-15.0,-15.0,-1.7,5.04,12.46,13.95,10.78],[-15.0,-15.0,-1.67,4.82,11.38,11.8,9.79],[-14.85,-15.0,-1.64,4.57,10.3,9.64,8.79],[-10.57,-14.47,-1.23,4.67,9.24,7.36,4.97],[-6.26,-12.1,-0.83,4.78,8.21,5.08,1.13],[-8.81,-10.69,-1.49,4.94,8.68,5.93,1.43],[-11.34,-9.27,-2.14,5.12,9.16,6.78,1.73],[-14.06,-8.77,-3.96,5.01,9.11,8.01,4.67],[-15.0,-8.26,-5.78,4.9,9.06,9.25,7.6],[-15.0,-7.59,-6.05,5.64,9.04,9.82,7.92],[-15.0,-6.92,-6.32,6.39,9.01,10.4,8.25],[-15.0,-7.36,-6.46,7.69,9.33,11.56,9.48],[-15.0,-7.82,-6.6,9.0,9.63,12.74,10.71],[-15.0,-7.41,-8.09,9.75,10.85,14.53,11.57],[-15.0,-7.01,-9.59,10.49,12.07,15.0,12.44],[-15.0,-4.63,-10.86,10.41,13.84,15.0,10.17],[-15.0,-2.21,-12.11,10.32,15.0,15.0,7.91],[-15.0,-2.17,-11.72,10.95,15.0,15.0,6.79],[-15.0,-2.11,-11.34,11.57,15.0,15.0,5.69],[-15.0,-3.82,-10.3,12.64,15.0,15.0,5.66],[-15.0,-5.52,-9.24,13.74,15.0,15.0,5.65],[-15.0,-5.52,-9.24,13.74,15.0,15.0,5.65],[-15.0,-5.52,-9.24,13.74,15.0,15.0,5.65],[-15.0,-5.52,-9.24,13.74,15.0,15.0,5.65]],"fill_db":-18.18222771416255,"skirt":0.2,"whine_db":[-35.0,-33.9,-40.3,-35.9,-46.0,-49.0]},"outside":{"noise_db":[-6.9,-14.6,-22.5,-4.8],"jit_db":[1.0,2.0,3.0,4.0],"eq_db":[-22.9,0.8,7.5,3.0,3.0],"corr_db":[[11.09,8.3,10.54,-6.92,-7.51,-7.96,-7.54],[11.09,8.3,10.54,-6.92,-7.51,-7.96,-7.54],[11.09,8.3,10.54,-6.92,-7.51,-7.96,-7.54],[11.09,8.3,10.54,-6.92,-7.51,-7.96,-7.54],[11.09,8.3,10.54,-6.92,-7.51,-7.96,-7.54],[11.09,8.3,10.54,-6.92,-7.51,-7.96,-7.54],[11.09,8.3,10.54,-6.92,-7.51,-7.96,-7.54],[11.09,8.3,10.54,-6.92,-7.51,-7.96,-7.54],[11.09,8.3,10.54,-6.92,-7.51,-7.96,-7.54],[11.09,8.3,10.54,-6.92,-7.51,-7.96,-7.54],[12.74,9.23,7.86,-6.51,-8.43,-7.66,-7.23],[14.4,10.16,5.18,-6.1,-9.34,-7.37,-6.92],[14.79,10.0,2.84,-6.21,-9.17,-6.42,-5.85],[15.0,9.83,0.53,-6.3,-9.0,-5.48,-4.76],[15.0,9.51,0.22,-5.95,-8.16,-5.51,-6.35],[15.0,9.19,-0.08,-5.6,-7.31,-5.51,-7.91],[15.0,9.76,0.2,-4.68,-6.55,-5.83,-11.05],[15.0,10.34,0.5,-3.78,-5.78,-6.14,-14.19],[15.0,10.48,0.73,-3.34,-5.05,-5.8,-15.0],[15.0,10.6,0.98,-2.9,-4.33,-5.45,-15.0],[15.0,10.32,1.12,-2.96,-4.26,-5.25,-15.0],[15.0,10.04,1.28,-2.99,-4.2,-5.05,-15.0],[15.0,10.64,0.91,-3.44,-4.77,-5.29,-15.0],[15.0,11.26,0.55,-3.89,-5.34,-5.56,-14.39],[15.0,11.59,0.12,-4.23,-5.53,-5.37,-12.99],[15.0,11.93,-0.27,-4.58,-5.71,-5.15,-11.6],[15.0,11.93,-0.27,-4.58,-5.71,-5.15,-11.6],[15.0,11.93,-0.27,-4.58,-5.71,-5.15,-11.6],[15.0,11.93,-0.27,-4.58,-5.71,-5.15,-11.6],[15.0,11.93,-0.27,-4.58,-5.71,-5.15,-11.6],[15.0,11.93,-0.27,-4.58,-5.71,-5.15,-11.6]],"fill_db":-2.038011626385507,"skirt":0.35,"whine_db":[-47.0,-45.9,-52.3,-55.9,-66.0,-69.0]},"distant":{"noise_db":[8.6,-9.6,-14.5,-40.0],"jit_db":[1.0,2.0,3.0,4.0],"eq_db":[-17.7,1.7,0.0,-22.7,-22.7],"corr_db":[[13.28,8.16,-0.47,-6.93,-8.22,2.19,-8.03],[13.28,8.16,-0.47,-6.93,-8.22,2.19,-8.03],[13.28,8.16,-0.47,-6.93,-8.22,2.19,-8.03],[13.28,8.16,-0.47,-6.93,-8.22,2.19,-8.03],[13.28,8.16,-0.47,-6.93,-8.22,2.19,-8.03],[13.28,8.16,-0.47,-6.93,-8.22,2.19,-8.03],[13.28,8.16,-0.47,-6.93,-8.22,2.19,-8.03],[13.28,8.16,-0.47,-6.93,-8.22,2.19,-8.03],[13.28,8.16,-0.47,-6.93,-8.22,2.19,-8.03],[13.28,8.16,-0.47,-6.93,-8.22,2.19,-8.03],[12.81,7.34,-1.23,-6.45,-6.72,2.06,-7.8],[12.33,6.51,-2.0,-5.99,-5.24,1.93,-7.56],[11.79,6.07,-2.47,-5.63,-4.92,0.9,-5.75],[11.24,5.63,-2.93,-5.29,-4.6,-0.13,-3.92],[10.99,4.69,-3.32,-5.64,-4.77,0.09,-2.05],[10.73,3.75,-3.7,-5.98,-4.93,0.31,-0.17],[10.22,3.55,-3.95,-6.47,-4.97,0.88,0.73],[9.71,3.33,-4.18,-6.93,-5.01,1.46,1.64],[7.77,3.79,-4.08,-6.95,-4.91,1.32,3.08],[5.82,4.25,-3.96,-6.97,-4.8,1.17,4.51],[5.64,3.63,-4.58,-6.94,-4.67,0.76,6.16],[5.44,3.02,-5.2,-6.91,-4.51,0.34,7.82],[6.76,1.62,-5.65,-6.58,-4.48,-0.12,8.47],[8.07,0.22,-6.09,-6.28,-4.46,-0.59,9.13],[7.58,-0.45,-5.71,-5.68,-4.43,-0.23,8.89],[7.11,-1.11,-5.33,-5.07,-4.4,0.14,8.67],[7.11,-1.11,-5.33,-5.07,-4.4,0.14,8.67],[7.11,-1.11,-5.33,-5.07,-4.4,0.14,8.67],[7.11,-1.11,-5.33,-5.07,-4.4,0.14,8.67],[7.11,-1.11,-5.33,-5.07,-4.4,0.14,8.67],[7.11,-1.11,-5.33,-5.07,-4.4,0.14,8.67]],"fill_db":-2.883968754545808,"skirt":0.35,"whine_db":[-35.0,-33.9,-40.3,-35.9,-46.0,-49.0]}},"air_k":1.0,"phi0":[-2.150145,1.998799,0.857663,1.351641,0.468724,2.479902,-0.613837,-2.10221,-1.375359,-2.729857,2.639848,-1.818368,2.466561,-1.494209,-1.662989,-2.38947,-1.597014,-0.761093,2.374037,1.542963,-0.99299,2.260009,-2.083914,0.80276,0.841098,1.03119,-0.235751,2.488923,-2.168293,-2.863789,0.902837,-1.756173,1.253041,-1.145915,1.771522,0.370704,-2.322847,2.674587,2.033011,2.713153,1.726686,1.656992,0.431411,0.005923,1.327167,-2.378236,-1.190772,0.576407,1.268731,-2.407795,2.914083,2.554671,0.674845,-0.568809,0.176393,-1.219457,2.299425,-1.388082,-0.387769,2.034725,-0.264186,1.498958,0.887992,2.071887,-1.318261,0.919811,2.738804,2.911759,0.14174,0.658534,1.138661,1.944285,-0.828366,0.111252,-1.751421,-1.664551,-0.886912,0.172288,-0.213914,0.261602,1.758194,2.734755,-0.058242,2.389322,0.355216,-0.620883,-0.512888,3.137597,-1.800079,1.691801,-2.630703,1.338136,-1.105952,-2.271576,1.823094,-0.580532]};
    var ORD = D.orders, CEN = D.centres, NO = ORD.length, NC = CEN.length, C0 = CEN[0], CSTEP = CEN[1] - CEN[0];
    var EQ_F = [100, 550, 1700, 5000, 11000];
    var OC = [0.670820, 1.414214, 2.828427, 5.656854, 11.313708, 22.627417, 39.395431];
    var FILL_O = [0.3, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75];
    var N_FILL = 8, N_WH = 6, N_NZ = 4;
    var NZ_F = [[50, 300], [300, 1000], [1000, 3000], [3000, 8000]];
    var TIMING_JIT = 0.35 * Math.PI / 180;
    function bqs(x, co, st) {
      var y = co[0] * x + st[0];
      st[0] = co[1] * x - co[3] * y + st[1];
      st[1] = co[2] * x - co[4] * y;
      return y;
    }
    function dbAmp(db) { return Math.pow(10, db / 20); }
    function ixRpm(rpm) {
      var r = rpm < C0 ? C0 : (rpm > CEN[NC - 1] ? CEN[NC - 1] : rpm);
      var t = (r - C0) / CSTEP, i0 = t | 0;
      if (i0 >= NC - 1) return { i0: NC - 2, i1: NC - 1, a: 1 };
      return { i0: i0, i1: i0 + 1, a: t - i0 };
    }
    function lerpArr(rows, ix, out) {
      var a = rows[ix.i0], b = rows[ix.i1], t = ix.a, i, n = a.length;
      if (!out) out = new Float32Array(n);
      for (i = 0; i < n; i++) out[i] = a[i] + (b[i] - a[i]) * t;
      return out;
    }
    function lerp1d(row, ix) { return row[ix.i0] + (row[ix.i1] - row[ix.i0]) * ix.a; }
    function logInterp(f, F, Y) {
      var lf = Math.log(Math.max(f, 20)), l0 = Math.log(F[0]), i, l1, t;
      if (lf <= l0) return Y[0];
      for (i = 1; i < F.length; i++) {
        l1 = Math.log(F[i]);
        if (lf <= l1) { t = (lf - l0) / (l1 - l0); return Y[i - 1] + t * (Y[i] - Y[i - 1]); }
        l0 = l1;
      }
      return Y[Y.length - 1];
    }
    function guard(row) {
      var n = row.length, tmp = new Float32Array(7), i, j, k, med, v, p;
      for (i = 0; i < n; i++) {
        k = 0;
        for (j = i - 3; j <= i + 3; j++) tmp[k++] = row[j < 0 ? 0 : (j >= n ? n - 1 : j)];
        for (j = 1; j < k; j++) {
          v = tmp[j]; p = j;
          while (p > 0 && tmp[p - 1] > v) { tmp[p] = tmp[p - 1]; p--; }
          tmp[p] = v;
        }
        med = tmp[3];
        if (row[i] < med - 12) row[i] = med - 12;
      }
    }
    function mulberry(seed) {
      var a = seed | 0;
      return function () {
        a = a + 0x6D2B79F5 | 0;
        var t = Math.imul(a ^ a >>> 15, 1 | a);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
      };
    }
    function groupOf(listener) {
      var p = (listener && (listener.position || listener.pos)) || 'rear';
      if (p === 'in_cabin' || p === 'onboard') return 'onboard';
      return 'paddock';
    }

    function BrmSynthV8(spec, listener, fs, opts) {
      this.fs = fs || 44100;
      this.spec = Object.assign({ idle_rpm: 3800, redline: 12000, name: 'BRM P30 V16' }, spec || {});
      this.rng = mulberry((opts && opts.seed) || 7);
      this.phRe = new Float32Array(NO);
      this.phIm = new Float32Array(NO);
      this.cw = new Float32Array(NO);
      this.sw = new Float32Array(NO);
      this.amp = new Float32Array(NO);
      this.ampT = new Float32Array(NO);
      this.live = new Int16Array(NO);
      this.nLive = 0;
      var i, p;
      for (i = 0; i < NO; i++) { p = D.phi0[i]; this.phRe[i] = Math.cos(p); this.phIm[i] = Math.sin(p); }
      this.whRe = new Float32Array(N_WH);
      this.whIm = new Float32Array(N_WH);
      this.whCw = new Float32Array(N_WH);
      this.whSw = new Float32Array(N_WH);
      this.whA = new Float32Array(N_WH);
      for (i = 0; i < N_WH; i++) { p = 0.7 * i; this.whRe[i] = Math.cos(p); this.whIm[i] = Math.sin(p); }
      this.fillSt = []; this.fillCo = [];
      for (i = 0; i < N_FILL; i++) { this.fillSt.push(new Float32Array(2)); this.fillCo.push(null); }
      this.nzSt = []; this.nzCo = []; this.nzG = new Float32Array(N_NZ); this.nzE = new Float32Array(N_NZ);
      for (i = 0; i < N_NZ; i++) { this.nzSt.push(new Float32Array(2)); this.nzCo.push(null); }
      this.airSt = new Float32Array(2); this.airCo = null;
      this.eRms = 1e-6; this.lvl = 0; this.thrS = 0; this.speed = 0;
      this.wfl = 1; this.jit = 0; this.skN = 0; this.skQ = 0;
      this.row = new Float32Array(NO); this.corr = new Float32Array(7); this.fillLv = new Float32Array(6);
      this.fLv = new Float32Array(N_FILL);
      this.fAmp = new Float32Array(N_FILL);
      this.setListener(listener);
      this._prepNz();
    }
    BrmSynthV8.prototype.setListener = function (listener) {
      this.group = groupOf(listener);
      this.rGain = this.group === 'onboard' ? 1 : 0.94;
    };
    BrmSynthV8.prototype.cloneWithListener = function (listener) {
      var c = new BrmSynthV8(this.spec, listener, this.fs);
      c.phRe.set(this.phRe); c.phIm.set(this.phIm);
      c.amp.set(this.amp); c.ampT.set(this.ampT);
      c.whRe.set(this.whRe); c.whIm.set(this.whIm);
      c.eRms = this.eRms; c.lvl = this.lvl; c.thrS = this.thrS;
      c.speed = this.speed; c.wfl = this.wfl;
      return c;
    };
    BrmSynthV8.prototype.ctl = function (m) {
      if (m && m.speed !== undefined) this.speed = m.speed;
    };
    BrmSynthV8.prototype._prepNz = function () {
      var i, mid, q, fs = this.fs, ny = 0.45 * fs, bw;
      for (i = 0; i < N_NZ; i++) {
        mid = Math.sqrt(NZ_F[i][0] * Math.min(NZ_F[i][1], ny));
        bw = Math.max(40, Math.min(NZ_F[i][1], ny) - NZ_F[i][0]);
        q = mid / bw; if (q < 0.5) q = 0.5; if (q > 4) q = 4;
        this.nzCo[i] = bq('bp', Math.min(mid, ny * 0.95), q, fs);
      }
      this.airCo = bq('bp', 2500, 0.55, fs);
    };
    BrmSynthV8.prototype._updateAmps = function (rpm, thr) {
      var g = this.group, cal = D.cal[g], ix = ixRpm(rpm), i, o, f, db, corr, inc, nL = 0, fs = this.fs, ny = 0.45 * fs;
      var row = lerpArr(D.rows[g], ix, this.row);
      guard(row);
      lerpArr(cal.corr_db, ix, this.corr);
      for (i = 0; i < NO; i++) {
        o = ORD[i];
        corr = logInterp(o, OC, this.corr);
        db = row[i] + corr;
        if (thr < 0.25) db += (1 - thr / 0.25) * D.ovr[i];
        f = o * rpm / 60 * (1 + this.jit * TIMING_JIT);
        db += logInterp(Math.max(f, 20), EQ_F, cal.eq_db);
        if (f >= ny || f < 8) { this.ampT[i] = 0; continue; }
        this.ampT[i] = dbAmp(db);
        inc = TWO_PI * f / fs;
        this.cw[i] = Math.cos(inc); this.sw[i] = Math.sin(inc);
        this.live[nL++] = i;
      }
      this.nLive = nL;
      lerpArr(D.fill[g], ix, this.fillLv);
      this.subLv = lerp1d(D.sub[g], ix);
      this.fillGain = dbAmp(cal.fill_db) * 8;
      this.cal = cal;
      this.eq100 = dbAmp(logInterp(100, EQ_F, cal.eq_db));
      this.fLv[0] = this.subLv;
      for (i = 0; i < 6; i++) this.fLv[i + 1] = this.fillLv[i];
      this.fLv[7] = this.fillLv[5] - 10;
      for (i = 0; i < N_FILL; i++) {
        f = FILL_O[i] * Math.max(rpm, 500) / 60;
        if (f < 30 || f > ny) this.fillCo[i] = null;
        else this.fillCo[i] = bq('bp', f, 2.4, fs);
      }
      var k, wamp;
      for (k = 0; k < N_WH; k++) {
        f = (k + 1) * D.whine * rpm / 60;
        if (f >= ny || f < 30) { this.whA[k] = 0; continue; }
        inc = TWO_PI * f / fs;
        this.whCw[k] = Math.cos(inc); this.whSw[k] = Math.sin(inc);
        this.whA[k] = dbAmp(cal.whine_db[k]) * 1.4142 * dbAmp(logInterp(f, EQ_F, cal.eq_db));
      }
    };
    BrmSynthV8.prototype.render = function (n, rpm, thr, L, R) {
      var i, ii, k, s, y, harm, body, nz, wh, air, a, re, im, cw, sw, runG, pr, lvlT, pow, scale, rel, env, tgt, lv, v;
      var fs = this.fs, ny = 0.45 * fs, cal, skirt, fillG, rng = this.rng;
      if (rpm < 80) rpm = 80;
      if (thr < 0) thr = 0; if (thr > 1) thr = 1;
      this.thrS += 0.08 * (thr - this.thrS);
      thr = this.thrS;
      runG = rpm < 400 ? 0 : (rpm < 2800 ? (rpm - 400) / 2400 : 1);
      this.jit += 0.02 * ((rng() * 2 - 1) - this.jit);
      this.skN += 0.04 * ((rng() * 2 - 1) - this.skN);
      this.skQ += 0.04 * ((rng() * 2 - 1) - this.skQ);
      this.wfl += 0.002 * ((rng() * 2 - 1) * 2.5 - (this.wfl - 1) * 8);
      this._updateAmps(rpm, thr);
      for (i = 0; i < NO; i++) this.amp[i] += 0.35 * (this.ampT[i] - this.amp[i]);
      pow = 0;
      this.nzE[0] = this.nzE[1] = this.nzE[2] = this.nzE[3] = 0;
      for (i = 0; i < NO; i++) {
        a = this.amp[i]; pow += a * a;
        var f = ORD[i] * rpm / 60, e = a * a * 0.5;
        if (f < 300) this.nzE[0] += e;
        else if (f < 1000) this.nzE[1] += e;
        else if (f < 3000) this.nzE[2] += e;
        else this.nzE[3] += e;
      }
      cal = this.cal; skirt = cal.skirt || 0;
      scale = Math.sqrt(Math.max(pow, 1e-12)) * this.eq100;
      fillG = this.fillGain * scale;
      for (i = 0; i < N_FILL; i++) this.fAmp[i] = this.fillCo[i] ? fillG * dbAmp(this.fLv[i]) : 0;
      pr = 1 + 4.7 * (rpm / 12000) * (rpm / 12000);
      lvlT = (0.25 + 0.75 * thr) * Math.pow(pr / 5.7, 0.4) * runG;
      this.lvl += 0.15 * (lvlT - this.lvl);
      rel = Math.max(0.4, Math.min(1, rpm / 6000));
      if (thr < 0.25) rel *= dbAmp(-2 * (1 - thr / 0.25));
      var wflA = dbAmp(3 * Math.max(-1, Math.min(1, this.wfl - 1)));
      var skN = this.skN, skQ = this.skQ, rGain = this.rGain, nLive = this.nLive;
      var airOn = this.group === 'onboard' && D.air_k > 0;
      var airK = D.air_k * Math.min((Math.max(this.speed, 0) / 50) * (Math.max(this.speed, 0) / 50), 2);
      for (i = 0; i < N_NZ; i++) {
        env = i === 3 ? Math.sqrt(pow * 0.5 + 1e-12) * 0.05 : Math.sqrt(this.nzE[i] + 1e-12);
        tgt = env * dbAmp(cal.noise_db[i]) * dbAmp(logInterp(Math.sqrt(NZ_F[i][0] * Math.min(NZ_F[i][1], ny)), EQ_F, cal.eq_db));
        this.nzG[i] += 0.08 * (tgt - this.nzG[i]);
      }
      for (s = 0; s < n; s++) {
        harm = 0;
        for (ii = 0; ii < nLive; ii++) {
          i = this.live[ii];
          a = this.amp[i];
          if (a < 1.2e-5) continue;
          cw = this.cw[i]; sw = this.sw[i];
          re = this.phRe[i] * cw - this.phIm[i] * sw;
          im = this.phRe[i] * sw + this.phIm[i] * cw;
          this.phRe[i] = re; this.phIm[i] = im;
          y = a * im;
          if (ORD[i] <= 3 && skirt > 0) y += a * skirt * (skN * im + skQ * re) * 0.7071;
          harm += y;
        }
        body = 0;
        for (i = 0; i < N_FILL; i++) {
          if (!this.fillCo[i]) continue;
          body += bqs((rng() * 2 - 1) * this.fAmp[i], this.fillCo[i], this.fillSt[i]);
        }
        nz = 0;
        for (i = 0; i < N_NZ; i++) nz += bqs((rng() * 2 - 1) * this.nzG[i] * 2.2, this.nzCo[i], this.nzSt[i]);
        this.eRms += 0.002 * (harm * harm - this.eRms);
        if (this.eRms < 1e-12) this.eRms = 1e-12;
        wh = 0;
        for (k = 0; k < N_WH; k++) {
          if (this.whA[k] <= 0) continue;
          cw = this.whCw[k]; sw = this.whSw[k];
          re = this.whRe[k] * cw - this.whIm[k] * sw;
          im = this.whRe[k] * sw + this.whIm[k] * cw;
          this.whRe[k] = re; this.whIm[k] = im;
          wh += this.whA[k] * im;
        }
        wh *= Math.sqrt(this.eRms) * rel * wflA;
        air = 0;
        if (airOn) air = bqs((rng() * 2 - 1), this.airCo, this.airSt) * airK * Math.sqrt(this.eRms);
        y = (harm + body + nz + wh + air) * this.lvl;
        L[s] = y * 0.55;
        R[s] = y * 0.55 * rGain;
      }
      for (i = 0; i < NO; i++) {
        a = this.phRe[i] * this.phRe[i] + this.phIm[i] * this.phIm[i];
        if (a > 0) { a = 1 / Math.sqrt(a); this.phRe[i] *= a; this.phIm[i] *= a; }
      }
      for (k = 0; k < N_WH; k++) {
        a = this.whRe[k] * this.whRe[k] + this.whIm[k] * this.whIm[k];
        if (a > 0) { a = 1 / Math.sqrt(a); this.whRe[k] *= a; this.whIm[k] *= a; }
      }
    };

    return [{ id: 'v8', Synth: BrmSynthV8, synthOpts: {}, positions: ['rear', 'in_cabin'] }];
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
    var DATA = /*V8P_DATA*/{"gt350":{"car":{"name":"1967 Shelby GT350 (289 Hi-Po)","bore":0.1016,"stroke":0.072898,"rod":0.130937,"cr":10.5,"fire_order":[1,5,4,2,6,3,7,8],"left_bank":[5,6,7,8],"adv_dur":306.0,"lsa":114.0,"ev_d":0.04064,"ev_lift":0.0116078,"camx":0.9,"lifters":"solid","idle_rpm":850.0,"redline":6800.0,"inertia":0.2,"carb":"1x4 Holley 715 cfm","th_area_wot":0.005771714879094934,"v_man":0.0026,"v_ac":0.006,"sn_len":0.05,"sn_d":0.1,"th_len":0.06,"exhaust":"tri-y","prim_len":0.55,"prim_d":0.038099999999999995,"sec_len":0.35,"sec_d":0.047625,"head_len":0.5,"pipe_d":0.05715,"x_len":0.3,"x_d":0.0508,"inter_len":1.2,"muf":[["chamber",0.45,6.0,0.35]],"tail_len":0.85,"bank_dl":0.12,"tri_pairs":[[1,2],[3,4],[5,7],[6,8]],"mouth":[[-0.05,0.3,0.3],[-0.05,-0.3,0.3]],"intake":[3.75,0.0,0.85],"covers":[[3.55,0.32,0.75],[3.55,-0.32,0.75]],"starter_pos":[3.2,-0.3,0.35],"head":[1.75,0.35,1.0],"cabin_len":1.9,"cabin_h":1.05,"st_ratio":15.7,"st_teeth":157,"st_free_rpm":4800.0,"st_i_free":80.0,"st_i_lock":550.0,"st_v_test":10.6,"r_batt":0.013,"st_kind":"ford_pe","v_batt":12.4,"st_i_sat":200.0,"spark0":12.0},"phys":{"q_comb":4.4528726927691595,"heat_loss":0.13964927768935703,"n_poly":1.32,"burn":61.12339112063403,"xr0":0.06,"xr_ov":0.03468261820807481,"cov":0.12291346956848595,"cov_res":0.9118891425544978,"mis_res":0.25,"cd":0.65,"eta_v":0.85,"wall":0.002,"hf":0.00013531634473443194,"ro_loss":0.98,"t_prim0":750.0,"t_prima":420.0,"k_int":1.4578821185363917,"r_th":0.6,"k_tick":1.3099889610031585,"k_jet":0.33401186308125413,"f_floor":15.0,"f_fw":72.41426333042916,"k_cab":0.35,"cab_q":4.312931693945894,"cab_model":1,"leak_floor":0.3,"leak_fw":0.003,"k_floor":1.000048686830272,"k_ind":0.029701259811889123,"f_abs":8000.0,"k_win":0.3,"fric0":1.347449491470554,"fric1":0.12,"fric2":0.018,"fric_cold":2.141082763671875,"a_idle":7.285542239127752e-05,"fast_idle":1.0,"prime_rev":0.9620153485230031,"catch_x":2.4,"react":0.3,"k_st":1.0,"k_hiss":24.59325797938332,"pack":0.06625559824334058,"bay_exh":0.8421933433992767,"k_nl":0.971139841482906,"dw_tau":0.4434368437152191,"gain":1.0}},"cadillac59":{"car":{"name":"1959 Cadillac Series 62 (390)","bore":0.1016,"stroke":0.098425,"rod":0.16827499999999998,"cr":10.5,"fire_order":[1,8,4,3,6,5,7,2],"left_bank":[1,3,5,7],"adv_dur":264.0,"lsa":114.0,"ev_d":0.039624,"ev_lift":0.01016,"camx":1.1,"lifters":"hydraulic","idle_rpm":520.0,"redline":5000.0,"inertia":0.45,"carb":"1x4 Carter AFB","th_area_wot":0.004979984443005094,"v_man":0.0032,"v_ac":0.009,"sn_len":0.12,"sn_d":0.075,"th_len":0.06,"exhaust":"log","port_len":0.12,"port_d":0.044449999999999996,"log_seg":0.11,"log_d":0.0508,"head_len":0.6,"pipe_d":0.0508,"x_len":0.25,"x_d":0.0508,"inter_len":1.5,"muf":[["chamber",0.25,8.0,0.0],["pipe",0.3,0.038099999999999995,0.0],["chamber",0.25,8.0,0.0],["pipe",0.35,0.0508,0.0],["chamber",0.35,5.0,0.1]],"tail_len":0.6,"bank_dl":0.25,"mouth":[[-0.1,0.55,0.45],[-0.1,-0.55,0.45]],"intake":[4.55,0.0,0.95],"covers":[[4.35,0.33,0.8],[4.35,-0.33,0.8]],"starter_pos":[3.95,0.3,0.35],"head":[2.3,0.4,1.05],"cabin_len":2.3,"cabin_h":1.05,"st_ratio":15.1,"st_teeth":166,"st_free_rpm":4350.0,"st_i_free":82.0,"st_i_lock":330.0,"st_v_test":10.6,"r_batt":0.012,"st_kind":"delco","v_batt":12.4,"st_i_sat":200.0,"spark0":5.0},"phys":{"q_comb":4.6,"heat_loss":0.08922799694585837,"n_poly":1.32,"burn":41.53321036001948,"xr0":0.06,"xr_ov":0.02,"cov":0.05032318942436391,"cov_res":0.4177863300939433,"mis_res":0.25,"cd":0.65,"eta_v":0.85,"wall":0.000965438109878401,"hf":0.0002345835862811804,"ro_loss":0.98,"t_prim0":731.0371827564371,"t_prima":420.0,"k_int":2.0,"r_th":0.6,"k_tick":1.1587881555704223,"k_jet":0.005,"f_floor":15.0,"f_fw":400.0,"k_cab":0.35,"cab_q":1.5,"cab_model":1,"leak_floor":0.3,"leak_fw":0.00529734107537258,"k_floor":1.003419309652263,"k_ind":0.03265282579623439,"f_abs":8000.0,"k_win":0.3,"fric0":1.8479475605609046,"fric1":0.12,"fric2":0.018,"fric_cold":1.345489501953125,"a_idle":5.1887408207057674e-05,"fast_idle":1.0,"prime_rev":0.7443756334908427,"catch_x":2.4,"react":0.3,"k_st":1.0,"k_hiss":6.541746651658922,"pack":1.7434902998281796,"bay_exh":0.4409732071662665,"k_nl":0.7884990991797303,"dw_tau":0.8054969365610245,"gain":1.0}},"p997":{"car":{"name":"2006 Porsche 911 Carrera 4S (997.1, M97.01 3.8)","bore":0.099,"stroke":0.0828,"rod":0.127,"cr":11.8,"fire_order":[1,6,2,4,3,5],"left_bank":[4,5,6],"adv_dur":252.0,"lsa":116.0,"ev_d":0.048648946545634474,"ev_lift":0.01,"camx":1.0,"lifters":"hydraulic","idle_rpm":670.0,"redline":7300.0,"inertia":0.16,"carb":"EFI, 1 electronic throttle body, two-stage resonance manifold","th_area_wot":0.004536459791783661,"v_man":0.0045,"v_ac":0.008,"sn_len":0.25,"sn_d":0.09,"th_len":0.1,"exhaust":"3-1","prim_len":0.3,"prim_x":[1.0,1.08,1.16],"prim_d":0.042,"head_len":0.25,"pipe_d":0.057,"x_len":0.0,"inter_len":0.6,"muf":[["chamber",0.3,4.0,0.15],["pipe",0.2,0.05,0.0],["chamber",0.22,10.0,0.5],["pipe",0.12,0.045,0.0],["chamber",0.22,10.0,0.5]],"tail_len":0.12,"bank_dl":0.04,"cross_mouth":true,"mouth":[[-0.02,0.52,0.3],[-0.02,-0.52,0.3]],"intake":[0.55,0.0,0.95],"covers":[[0.75,0.42,0.42],[0.75,-0.42,0.42]],"starter_pos":[1.3,0.22,0.35],"head":[2.15,0.35,1.0],"cabin_len":1.9,"cabin_h":1.0,"rear_engine":true,"eng_shadow":0.8,"st_ratio":54.888888888888886,"st_teeth":130,"st_free_rpm":16000.0,"st_i_free":70.0,"st_i_lock":850.0,"st_v_test":11.5,"st_v_lock":6.0,"r_batt":0.008,"st_kind":"bosch_pm","v_batt":12.5,"st_i_sat":10000.0,"spark0":10.0},"phys":{"q_comb":4.6,"heat_loss":0.06610091648405512,"n_poly":1.32,"burn":59.99578391306203,"xr0":0.05,"xr_ov":0.02,"cov":0.06304950659730628,"cov_res":0.07350737085237899,"mis_res":0.25,"cd":0.65,"eta_v":0.85,"wall":0.001324638794939649,"hf":0.0016656077186669369,"ro_loss":0.98,"t_prim0":678.3541752286684,"t_prima":420.0,"k_int":2.0,"r_th":0.6,"k_tick":0.0,"k_jet":0.27672619972869567,"f_floor":15.0,"f_fw":400.0,"k_cab":0.35,"cab_q":2.6669410585477147,"cab_model":1,"leak_floor":0.3,"leak_fw":0.003,"k_floor":0.9981515491453153,"k_ind":0.5,"f_abs":8000.0,"k_win":0.3,"fric0":1.0197330664957027,"fric1":0.12,"fric2":0.018,"fric_cold":1.531341552734375,"a_idle":3.5e-05,"fast_idle":1.0,"prime_rev":2.8,"catch_x":2.4,"react":0.3,"k_st":1.0,"k_hiss":40.0,"pack":2.227712131406225,"bay_exh":0.17202969592712633,"k_nl":0.9301919446001485,"dw_tau":0.10000000000000002,"gain":1.0}}}/*END_V8P_DATA*/;
    var P_ATM = 101325.0, R_GAS = 287.0, G_EX = 1.30, G_AIR = 1.40, T_AMB = 293.0, C_AIR = 343.0, RHO_AIR = 1.2;
    var DL = 1024, DLM = DL - 1, HB = 4096, NSRC = 6;
    // Q
    var Q_BORE_A = 0, Q_CRANK_R = 1, Q_ROD = 2, Q_VC = 3, Q_IVO = 4, Q_IVC = 5, Q_EVO = 6, Q_EVC = 7, Q_DV = 8, Q_LMAX = 9, Q_CAMX = 10,
      Q_CD = 11, Q_QC = 12, Q_HL = 13, Q_NP = 14, Q_BURN = 15, Q_XR0 = 16, Q_XROV = 17, Q_OVL = 18, Q_COV = 19, Q_COVR = 20, Q_MISR = 21,
      Q_ETAV = 22, Q_VD = 23, Q_VMAN = 24, Q_ATHW = 25, Q_AIDLE = 26, Q_FIDLE = 27, Q_ROL = 28, Q_J = 29, Q_F0 = 30, Q_F1 = 31, Q_F2 = 32,
      Q_FCOLD = 33, Q_STG = 34, Q_STKE = 35, Q_STRM = 36, Q_STRB = 37, Q_STIF = 38, Q_STJM = 39, Q_TEETH = 40, Q_VBAT = 41, Q_SOLID = 42,
      Q_KJET = 43, Q_VAC = 44, Q_LTH = 45, Q_RTH = 46, Q_SNM = 47, Q_SNR = 48, Q_SPARK0 = 49, Q_TAILA = 50, Q_NSEG = 51, Q_DWT = 52, Q_KNL = 53, Q_STIS = 54, NQ = 56;
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
          if (phi >= Q[Q_IVO] && phi < Q[Q_IVC]) qin += dvd * w;
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
    function camEvents(c) {
      var d = c.adv_dur / 2.0;
      return [360.0 + (c.lsa - d), 360.0 + (c.lsa + d), 360.0 - (c.lsa + d), 360.0 - (c.lsa - d)];
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
          var d = mf[0] === 'chamber' ? c.pipe_d * Math.sqrt(mf[2]) : mf[2];
          var p = pipe(mf[1], d, 2, mf[3]); node(0, 0, [[prev, 1], [p, 0]]); prev = p;
        });
        var tail = pipe(c.tail_len, c.pipe_d, 2);
        node(0, 0, [[prev, 1], [tail, 0]]);
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
        this.PF[f + F_GAIN] = Math.exp(-P.wall * L / d);
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
        var ch = Math.exp(-Math.max(tcat, 0.0) / 14.0);
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

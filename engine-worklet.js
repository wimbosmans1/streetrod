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
    settling() { return !!(this.cur.settling && this.cur.settling()); }   // v1.11.43: silent pre-roll still running
    bay(o) { return this.cur.bay ? this.cur.bay(o) : false; }   // v1.10 engine bay: kernel crank state if the model has one
    ownNoise() { return !!(this.cur.ownNoise && this.cur.ownNoise()); }   // v19: the source renders its own tyre / wind noise
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
        // v1.11.43: hold the crossfade while the new source is still in its silent pre-roll (no level dip)
      }
      if (this.nsrc && !(this.nsrc.settling && this.nsrc.settling())) {
        const g2 = this.nsrc.outGain, step = 1 / (0.25 * this.fs), sgB = !!this.nsrc.selfGated;
        for (let k = 0; k < n; k++) {
          this.xfade = Math.min(1, this.xfade + step);
          const a = Math.cos(this.xfade * Math.PI / 2), b = Math.sin(this.xfade * Math.PI / 2), gb = g2 * (sgB ? 1 : ge[k]);
          tL[k] = a * tL[k] + b * gb * this.tL2[k]; tR[k] = a * tR[k] + b * gb * this.tR2[k];
        }
        if (this.xfade >= 1) { this.src = this.nsrc; this.outGain = this.nsrc.outGain; this.modelId = this.nsrc.modelId; this.nsrc = null; }
      }
      const fs = this.fs, ownN = !!(this.src.ownNoise && this.src.ownNoise());
      for (let k = 0; k < n; k++) {
        this.starterLvl += ((this.src.selfGated ? 0 : this.starterTarget) - this.starterLvl) * this.gainK;
        this.speedS += (this.speed - this.speedS) * 0.0005; this.skidS += (this.skid - this.skidS) * 0.002;
        const w = wv[k];
        const eL = tL[k], eR = tR[k];
        const v = this.speedS, vr = v / 40;
        const road = ownN ? 0 : Math.min(0.08, 0.05 * vr * vr);   // v19 sets render their own sourced road / wind noise (skid stays here)
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
  EngineVoice.prototype.dspStats = function () {
    const s = this.nsrc || this.src;
    const syn = s && s.cur ? s.cur : s;
    if (!syn) return { nan: 0, pipes: 0, pipes0: 0, lite: false };
    return { nan: syn.nanResets || 0, pipes: syn.NP || 0, pipes0: syn.NP0 || syn.NP || 0, lite: !!syn.netLite };
  };
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
        this.n = 0; this.overruns = 0; this.cpuAcc = 0; this.cpu = 0;
        this.port.postMessage({ type: 'hello', fs: sampleRate });
      }
      process(inputs, outputs) {
        const out = outputs[0]; const L = out[0], R = out[1] || out[0];
        var t0 = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
        this.voice.process(L, R, L.length);
        var t1 = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
        this.cpuAcc += (t1 - t0);
        var budget = 1000 * L.length / sampleRate;
        // Accumulate only time spent inside process() (not the gaps between quanta).
        if ((++this.n & 31) === 0) {
          var audioMs = 32 * budget;
          this.cpu = audioMs > 0 ? this.cpuAcc / audioMs : 0;
          if (this.cpu > 1) this.overruns++;
          this.cpuAcc = 0;
          var st = this.voice.dspStats();
          this.port.postMessage({ type: 'meter', gr: this.voice.gr, blocks: this.n, model: this.voice.modelId, err: this.voice.error, cpu: this.cpu, overruns: this.overruns, nan: st.nan, pipes: st.pipes, pipes0: st.pipes0, lite: st.lite });
        }
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
    var DATA = /*V8P_DATA*/{"gt350":{"car":{"name":"1967 Shelby GT350 (289 Hi-Po)","bore":0.1016,"stroke":0.072898,"rod":0.130937,"cr":10.5,"fire_order":[1,5,4,2,6,3,7,8],"left_bank":[5,6,7,8],"adv_dur":306,"lsa":114,"ev_d":0.04064,"ev_lift":0.0116078,"camx":0.9,"lifters":"solid","idle_rpm":850,"redline":6800,"inertia":0.2,"carb":"1x4 Holley 715 cfm","th_area_wot":0.005771714879094934,"v_man":0.0026,"v_ac":0.006,"sn_len":0.05,"sn_d":0.1,"th_len":0.06,"exhaust":"tri-y","prim_len":0.55,"prim_d":0.038099999999999995,"sec_len":0.35,"sec_d":0.047625,"head_len":0.5,"pipe_d":0.05715,"x_len":0.3,"x_d":0.0508,"inter_len":1.2,"muf":[["chamber",0.45,6,0.35]],"tail_len":0.85,"bank_dl":0.12,"tri_pairs":[[1,2],[3,4],[5,7],[6,8]],"mouth":[[-0.05,0.3,0.3],[-0.05,-0.3,0.3]],"intake":[3.75,0,0.85],"covers":[[3.55,0.32,0.75],[3.55,-0.32,0.75]],"starter_pos":[3.2,-0.3,0.35],"head":[1.75,0.35,1],"cabin_len":1.9,"cabin_h":1.05,"st_ratio":15.7,"st_teeth":157,"st_free_rpm":4800,"st_i_free":80,"st_i_lock":550,"st_v_test":10.6,"r_batt":0.013,"st_kind":"ford_pe","v_batt":12.4,"st_i_sat":200,"spark0":12},"phys":{"q_comb":4.4528726927691595,"heat_loss":0.13964927768935703,"n_poly":1.32,"burn":61.12339112063403,"xr0":0.06,"xr_ov":0.03468261820807481,"cov":0.12291346956848595,"cov_res":0.9118891425544978,"mis_res":0.25,"cd":0.65,"eta_v":0.85,"wall":0.002,"hf":0.00013531634473443194,"ro_loss":0.98,"t_prim0":750,"t_prima":420,"k_int":1.4578821185363917,"r_th":0.6,"k_tick":1.3099889610031585,"k_jet":0.33401186308125413,"f_floor":15,"f_fw":72.41426333042916,"k_cab":0.35,"cab_q":4.312931693945894,"cab_model":1,"leak_floor":0.3,"leak_fw":0.003,"k_floor":1.000048686830272,"k_ind":0.029701259811889123,"f_abs":8000,"k_win":0.3,"fric0":1.347449491470554,"fric1":0.12,"fric2":0.018,"fric_cold":2.14,"a_idle":7.285542239127752e-05,"fast_idle":1,"prime_rev":0.6,"catch_x":2.4,"react":0.3,"k_st":1,"k_hiss":24.59325797938332,"pack":0.06625559824334058,"bay_exh":0.8421933433992767,"k_nl":0.971139841482906,"dw_tau":0.4434368437152191,"gain":1,"k_st_mesh":0.38,"k_st_comm":0.2,"k_st_mot":0.16,"k_st_bb":1,"k_st_out":1,"k_st_cab":1.28,"k_st_bb2":0.5,"k_st_bbhi":0.22,"k_st_am":0.55,"k_st_am_far":0.58,"k_st_am_out":0.5,"k_st_am_cab":0.52,"st_lead":0.04}},"cadillac59":{"car":{"name":"1959 Cadillac Series 62 (390)","bore":0.1016,"stroke":0.098425,"rod":0.16827499999999998,"cr":10.5,"fire_order":[1,8,4,3,6,5,7,2],"left_bank":[1,3,5,7],"adv_dur":264,"lsa":114,"ev_d":0.039624,"ev_lift":0.01016,"camx":1.1,"lifters":"hydraulic","idle_rpm":520,"redline":5000,"inertia":0.45,"carb":"1x4 Carter AFB","th_area_wot":0.004979984443005094,"v_man":0.0032,"v_ac":0.009,"sn_len":0.12,"sn_d":0.075,"th_len":0.06,"exhaust":"log","port_len":0.12,"port_d":0.044449999999999996,"log_seg":0.11,"log_d":0.0508,"head_len":0.6,"pipe_d":0.0508,"x_len":0.25,"x_d":0.0508,"inter_len":1.5,"muf":[["chamber",0.25,8,0],["pipe",0.3,0.038099999999999995,0],["chamber",0.25,8,0],["pipe",0.35,0.0508,0],["chamber",0.35,5,0.1]],"tail_len":0.6,"bank_dl":0.25,"mouth":[[-0.1,0.55,0.45],[-0.1,-0.55,0.45]],"intake":[4.55,0,0.95],"covers":[[4.35,0.33,0.8],[4.35,-0.33,0.8]],"starter_pos":[3.95,0.3,0.35],"head":[2.3,0.4,1.05],"cabin_len":2.3,"cabin_h":1.05,"st_ratio":15.1,"st_teeth":166,"st_free_rpm":4350,"st_i_free":82,"st_i_lock":330,"st_v_test":10.6,"r_batt":0.012,"st_kind":"delco","v_batt":12.4,"st_i_sat":200,"spark0":5},"phys":{"q_comb":4.6,"heat_loss":0.08922799694585837,"n_poly":1.32,"burn":41.53321036001948,"xr0":0.06,"xr_ov":0.02,"cov":0.05032318942436391,"cov_res":0.4177863300939433,"mis_res":0.25,"cd":0.65,"eta_v":0.85,"wall":0.000965438109878401,"hf":0.0002345835862811804,"ro_loss":0.98,"t_prim0":731.0371827564371,"t_prima":420,"k_int":2,"r_th":0.6,"k_tick":1.1587881555704223,"k_jet":0.005,"f_floor":15,"f_fw":400,"k_cab":0.35,"cab_q":1.5,"cab_model":1,"leak_floor":0.3,"leak_fw":0.00529734107537258,"k_floor":1.003419309652263,"k_ind":0.03265282579623439,"f_abs":8000,"k_win":0.3,"fric0":1.8479475605609046,"fric1":0.12,"fric2":0.018,"fric_cold":1.2,"a_idle":5.1887408207057674e-05,"fast_idle":1,"prime_rev":0.7443756334908427,"catch_x":2.4,"react":0.3,"k_st":1,"k_hiss":6.541746651658922,"pack":1.7434902998281796,"bay_exh":0.4409732071662665,"k_nl":0.7884990991797303,"dw_tau":0.8054969365610245,"gain":1,"k_st_mesh":0.42,"k_st_comm":0.16,"k_st_mot":0.14,"k_st_bb":0.9,"k_st_out":1,"k_st_cab":1.2,"k_st_bb2":0.48,"k_st_bbhi":0.22,"k_st_am":0.5,"k_st_am_far":0.55,"k_st_am_out":0.48,"k_st_am_cab":0.48,"st_lead":0.03},"sets":{"p2":{"q_comb":4.6,"heat_loss":0.08922799694585837,"n_poly":1.32,"burn":41.53321036001948,"xr0":0.06,"xr_ov":0.02,"cov":0.05032318942436391,"cov_res":0.4177863300939433,"mis_res":0.25,"cd":0.65,"eta_v":0.85,"wall":0.000965438109878401,"hf":0.0002345835862811804,"ro_loss":0.98,"t_prim0":731.0371827564371,"t_prima":420,"k_int":2,"r_th":0.6,"k_tick":1.1587881555704223,"k_jet":0.005,"f_floor":15,"f_fw":400,"k_cab":0.35,"cab_q":1.5,"cab_model":1,"leak_floor":0.3,"leak_fw":0.00529734107537258,"k_floor":1.003419309652263,"k_ind":0.03265282579623439,"f_abs":8000,"k_win":0.3,"fric0":1.8479475605609046,"fric1":0.12,"fric2":0.018,"fric_cold":1.2,"a_idle":5.1887408207057674e-05,"fast_idle":1,"prime_rev":0.7443756334908427,"catch_x":2.4,"react":0.3,"k_st":1,"k_hiss":6.541746651658922,"pack":1.7434902998281796,"bay_exh":0.4409732071662665,"k_nl":0.7884990991797303,"dw_tau":0.8054969365610245,"gain":1,"k_st_mesh":0.42,"k_st_comm":0.16,"k_st_mot":0.14,"k_st_bb":0.9,"k_st_out":1,"k_st_cab":1.2,"k_st_bb2":0.48,"k_st_bbhi":0.22,"k_st_am":0.5,"k_st_am_far":0.55,"k_st_am_out":0.48,"k_st_am_cab":0.48,"st_lead":0.03,"rod":0.1651,"ev_d":0.038099999999999995,"ev_lift":0.01143,"adv_dur":282.0,"lsa":114.0,"in_dur":282.0,"in_lsa":114.0,"ex_dur":282.0,"ex_lsa":112.0,"x_len":0.0,"tail_d":0.044449999999999996,"v_ac":0.009163954184270904,"sn_d":0.1016,"sn_len":0.1016,"muf":[["chamber",0.25,8.0,0.0],["pipe",0.3,0.038099999999999995,0.0],["chamber",0.25,8.0,0.0],["pipe",0.35,0.044449999999999996,0.0],["chamber",0.35,5.0,0.1]]},"p3":{"q_comb":4.6,"heat_loss":0.08922799694585837,"n_poly":1.32,"burn":41.53321036001948,"xr0":0.06,"xr_ov":0.02,"cov":0.05032318942436391,"cov_res":0.4177863300939433,"mis_res":0.25,"cd":0.65,"eta_v":0.85,"wall":0.000965438109878401,"hf":0.0002345835862811804,"ro_loss":0.98,"t_prim0":731.0371827564371,"t_prima":420,"k_int":2,"r_th":0.6,"k_tick":1.1587881555704223,"k_jet":0.005,"f_floor":15,"f_fw":400,"k_cab":0.35,"cab_q":1.5,"cab_model":1,"leak_floor":0.3,"leak_fw":0.00529734107537258,"k_floor":1.003419309652263,"k_ind":0.03265282579623439,"f_abs":8000,"k_win":0.3,"fric0":1.8479475605609046,"fric1":0.12,"fric2":0.018,"fric_cold":1.2,"a_idle":5.1887408207057674e-05,"fast_idle":1,"prime_rev":0.7443756334908427,"catch_x":2.4,"react":0.3,"k_st":1,"k_hiss":6.541746651658922,"pack":1.7434902998281796,"bay_exh":0.4409732071662665,"k_nl":0.7884990991797303,"dw_tau":0.8054969365610245,"gain":1,"k_st_mesh":0.42,"k_st_comm":0.16,"k_st_mot":0.14,"k_st_bb":0.9,"k_st_out":1,"k_st_cab":1.2,"k_st_bb2":0.48,"k_st_bbhi":0.22,"k_st_am":0.5,"k_st_am_far":0.55,"k_st_am_out":0.48,"k_st_am_cab":0.48,"st_lead":0.03,"rod":0.1651,"ev_d":0.038099999999999995,"ev_lift":0.01143,"adv_dur":282.0,"lsa":114.0,"in_dur":282.0,"in_lsa":114.0,"ex_dur":282.0,"ex_lsa":112.0,"x_len":0.0,"tail_d":0.044449999999999996,"v_ac":0.009163954184270904,"sn_d":0.1016,"sn_len":0.1016,"muf":[["pipe",0.3857625,0.0508,0.0],["chamber",0.1285875,6.296875000000001,0.0],["perf",0.257175,0.044449999999999996,0.257175,0.12747536036426804,0.2,0.0,3],["chamber",0.1285875,6.296875000000001,0.0],["pipe",0.3857625,0.044449999999999996,0.0],["cavity",0.1285875,0.12747536036426804,0.0,0.0508,0.038099999999999995,0.02,0.0],["pipe",0.35,0.044449999999999996,0.0],["perf",0.5715,0.044449999999999996,0.5715,0.13017499999999999,0.2,0.0,3]]},"p4":{"q_comb":4.6,"heat_loss":0.08922799694585837,"n_poly":1.32,"burn":41.53321036001948,"xr0":0.06,"xr_ov":0.02,"cov":0.05032318942436391,"cov_res":0.4177863300939433,"mis_res":0.25,"cd":0.65,"eta_v":0.85,"wall":0.000965438109878401,"hf":0.0002345835862811804,"ro_loss":0.98,"t_prim0":731.0371827564371,"t_prima":420,"k_int":2,"r_th":0.6,"k_tick":1.1587881555704223,"k_jet":0.005,"f_floor":15,"f_fw":400,"k_cab":0.35,"cab_q":1.5,"cab_model":1,"leak_floor":0.3,"leak_fw":0.00529734107537258,"k_floor":1.003419309652263,"k_ind":0.03265282579623439,"f_abs":8000,"k_win":0.3,"fric0":1.8479475605609046,"fric1":0.12,"fric2":0.018,"fric_cold":1.2,"a_idle":5.1887408207057674e-05,"fast_idle":1,"prime_rev":0.7443756334908427,"catch_x":2.4,"react":0.3,"k_st":1,"k_hiss":6.541746651658922,"pack":1.7434902998281796,"bay_exh":0.4409732071662665,"k_nl":0.7884990991797303,"dw_tau":0.8054969365610245,"gain":1,"k_st_mesh":0.42,"k_st_comm":0.16,"k_st_mot":0.14,"k_st_bb":0.9,"k_st_out":1,"k_st_cab":1.2,"k_st_bb2":0.48,"k_st_bbhi":0.22,"k_st_am":0.5,"k_st_am_far":0.55,"k_st_am_out":0.48,"k_st_am_cab":0.48,"st_lead":0.03,"rod":0.1651,"ev_d":0.038099999999999995,"ev_lift":0.01143,"adv_dur":282.0,"lsa":114.0,"in_dur":282.0,"in_lsa":114.0,"ex_dur":282.0,"ex_lsa":112.0,"x_len":0.0,"tail_d":0.044449999999999996,"v_ac":0.009163954184270904,"sn_d":0.1016,"sn_len":0.1016,"muf":[["pipe",0.3857625,0.0508,0.0],["cavity",0.1285875,0.12747536036426804,0.0,0.0508,0.0508,0.02,0.0],["perf",0.257175,0.044449999999999996,0.257175,0.12747536036426804,0.2,0.0,3,0.0015,0.003175,0.13],["cavity",0.1285875,0.12747536036426804,0.0,0.0508,0.044449999999999996,0.02,0.0],["pipe",0.3857625,0.044449999999999996,0.0],["cavity",0.1285875,0.12747536036426804,0.0,0.0508,0.038099999999999995,0.02,0.0],["pipe",0.35,0.044449999999999996,0.0],["perf",0.5715,0.044449999999999996,0.5715,0.13017499999999999,0.2,0.0,3,0.0015,0.003175,0.13]]},"p5":{"q_comb":4.6,"heat_loss":0.08922799694585837,"n_poly":1.32,"burn":41.53321036001948,"xr0":0.06,"xr_ov":0.02,"cov":0.05032318942436391,"cov_res":0.4177863300939433,"mis_res":0.25,"cd":0.65,"eta_v":0.85,"wall":0.000965438109878401,"hf":0.0002345835862811804,"ro_loss":0.98,"t_prim0":731.0371827564371,"t_prima":420,"k_int":2,"r_th":0.6,"k_tick":1.1587881555704223,"k_jet":0.005,"f_floor":15,"f_fw":400,"k_cab":0.35,"cab_q":1.5,"cab_model":1,"leak_floor":0.3,"leak_fw":0.00529734107537258,"k_floor":1.003419309652263,"k_ind":0.03265282579623439,"f_abs":8000,"k_win":0.3,"fric0":1.8479475605609046,"fric1":0.12,"fric2":0.018,"fric_cold":1.2,"a_idle":5.1887408207057674e-05,"fast_idle":1,"prime_rev":0.7443756334908427,"catch_x":2.4,"react":0.3,"k_st":1,"k_hiss":6.541746651658922,"pack":1.7434902998281796,"bay_exh":0.4409732071662665,"k_nl":0.7884990991797303,"dw_tau":0.8054969365610245,"gain":1,"k_st_mesh":0.42,"k_st_comm":0.16,"k_st_mot":0.14,"k_st_bb":0.9,"k_st_out":1,"k_st_cab":1.2,"k_st_bb2":0.48,"k_st_bbhi":0.22,"k_st_am":0.5,"k_st_am_far":0.55,"k_st_am_out":0.48,"k_st_am_cab":0.48,"st_lead":0.03,"rod":0.1651,"ev_d":0.038099999999999995,"ev_lift":0.01143,"adv_dur":282.0,"lsa":114.0,"in_dur":282.0,"in_lsa":114.0,"ex_dur":282.0,"ex_lsa":112.0,"x_len":0.0,"tail_d":0.044449999999999996,"v_ac":0.009163954184270904,"sn_d":0.1016,"sn_len":0.1016,"muf":[["pipe",0.3857625,0.0508,0.0],["cavity",0.1285875,0.12747536036426804,0.0,0.0508,0.0508,0.02,0.0],["perf",0.257175,0.044449999999999996,0.257175,0.12747536036426804,0.2,0.0,3,0.0015,0.003175,0.13],["cavity",0.1285875,0.12747536036426804,0.0,0.0508,0.044449999999999996,0.02,0.0],["pipe",0.3857625,0.044449999999999996,0.0],["cavity",0.1285875,0.12747536036426804,0.0,0.0508,0.038099999999999995,0.02,0.0],["pipe",3.07,0.044449999999999996,0.0],["perf",0.5715,0.044449999999999996,0.5715,0.13017499999999999,0.2,0.0,3,0.0015,0.003175,0.13]],"tail_len":0.5,"cabin_len":2.19,"cabin_h":1.11},"p6":{"q_comb":4.6,"heat_loss":0.08922799694585837,"n_poly":1.32,"burn":41.53321036001948,"xr0":0.06,"xr_ov":0.02,"cov":0.05032318942436391,"cov_res":0.4177863300939433,"mis_res":0.25,"cd":0.65,"eta_v":0.85,"wall":0.000965438109878401,"hf":0.0002345835862811804,"ro_loss":0.98,"t_prim0":731.0371827564371,"t_prima":420,"k_int":2,"r_th":0.6,"k_tick":1.1587881555704223,"k_jet":0.005,"f_floor":15,"f_fw":400,"k_cab":0.35,"cab_q":1.5,"cab_model":1,"leak_floor":0.3,"leak_fw":0.00529734107537258,"k_floor":1.003419309652263,"k_ind":0.03265282579623439,"f_abs":8000,"k_win":0.3,"fric0":1.8479475605609046,"fric1":0.12,"fric2":0.018,"fric_cold":1.2,"a_idle":5.1887408207057674e-05,"fast_idle":1,"prime_rev":0.7443756334908427,"catch_x":2.4,"react":0.3,"k_st":1,"k_hiss":6.541746651658922,"pack":1.7434902998281796,"bay_exh":0.4409732071662665,"k_nl":0.7884990991797303,"dw_tau":0.8054969365610245,"gain":1,"k_st_mesh":0.42,"k_st_comm":0.16,"k_st_mot":0.14,"k_st_bb":0.9,"k_st_out":1,"k_st_cab":1.2,"k_st_bb2":0.48,"k_st_bbhi":0.22,"k_st_am":0.5,"k_st_am_far":0.55,"k_st_am_out":0.48,"k_st_am_cab":0.48,"st_lead":0.03,"rod":0.1651,"ev_d":0.038099999999999995,"ev_lift":0.01143,"adv_dur":282.0,"lsa":114.0,"in_dur":282.0,"in_lsa":114.0,"ex_dur":282.0,"ex_lsa":112.0,"x_len":0.0,"tail_d":0.044449999999999996,"v_ac":0.009163954184270904,"sn_d":0.1016,"sn_len":0.1016,"muf":[["pipe",0.3857625,0.0508,0.0],["cavity",0.1285875,0.12747536036426804,2.0,0.0508,0.0508,0.02,0.0],["perf",0.257175,0.044449999999999996,0.257175,0.12747536036426804,0.2,2.0,3,0.0015,0.003175,2.0],["cavity",0.1285875,0.12747536036426804,2.0,0.0508,0.044449999999999996,0.02,0.0],["pipe",0.3857625,0.044449999999999996,0.0],["cavity",0.1285875,0.12747536036426804,2.0,0.0508,0.038099999999999995,0.02,0.0],["pipe",3.07,0.044449999999999996,0.0],["perf",0.5715,0.044449999999999996,0.5715,0.13017499999999999,0.2,2.0,3,0.0015,0.003175,2.0]],"tail_len":0.5,"cabin_len":2.19,"cabin_h":1.11},"p7":{"q_comb":4.6,"heat_loss":0.08922799694585837,"n_poly":1.32,"burn":41.53321036001948,"xr0":0.06,"xr_ov":0.02,"cov":0.05032318942436391,"cov_res":0.4177863300939433,"mis_res":0.25,"cd":0.65,"eta_v":0.85,"wall":0.000965438109878401,"hf":0.0002345835862811804,"ro_loss":0.98,"t_prim0":731.0371827564371,"t_prima":420,"k_int":2,"r_th":0.6,"k_tick":1.1587881555704223,"k_jet":0.005,"f_floor":15,"f_fw":400,"k_cab":0.35,"cab_q":1.5,"cab_model":1,"leak_floor":0.3,"leak_fw":0.00529734107537258,"k_floor":1.003419309652263,"k_ind":0.03265282579623439,"f_abs":8000,"k_win":0.3,"fric0":1.8479475605609046,"fric1":0.12,"fric2":0.018,"fric_cold":1.2,"a_idle":5.1887408207057674e-05,"fast_idle":1,"prime_rev":0.7443756334908427,"catch_x":2.4,"react":0.3,"k_st":1,"k_hiss":6.541746651658922,"pack":1.7434902998281796,"bay_exh":0.4409732071662665,"k_nl":0.7884990991797303,"dw_tau":0.8054969365610245,"gain":1,"k_st_mesh":0.42,"k_st_comm":0.16,"k_st_mot":0.14,"k_st_bb":0.9,"k_st_out":1,"k_st_cab":1.2,"k_st_bb2":0.48,"k_st_bbhi":0.22,"k_st_am":0.5,"k_st_am_far":0.55,"k_st_am_out":0.48,"k_st_am_cab":0.48,"st_lead":0.03,"rod":0.1651,"ev_d":0.038099999999999995,"ev_lift":0.01143,"adv_dur":282.0,"lsa":114.0,"in_dur":282.0,"in_lsa":114.0,"ex_dur":282.0,"ex_lsa":112.0,"x_len":0.0,"tail_d":0.044449999999999996,"v_ac":0.009163954184270904,"sn_d":0.1016,"sn_len":0.1016,"muf":[["pipe",0.3857625,0.0508,0.0],["cavity",0.1285875,0.12747536036426804,2.0,0.0508,0.0508,0.02,0.0],["perf",0.257175,0.044449999999999996,0.257175,0.12747536036426804,0.2,2.0,3,0.0015,0.003175,2.0],["cavity",0.1285875,0.12747536036426804,2.0,0.0508,0.044449999999999996,0.02,0.0],["pipe",0.3857625,0.044449999999999996,0.0],["cavity",0.1285875,0.12747536036426804,2.0,0.0508,0.038099999999999995,0.02,0.0],["pipe",3.07,0.044449999999999996,0.0],["perf",0.5715,0.044449999999999996,0.5715,0.13017499999999999,0.2,2.0,3,0.0015,0.003175,2.0]],"tail_len":0.5,"cabin_len":2.19,"cabin_h":1.11,"r_ac":3900.0},"v13":{"q_comb":4.6,"heat_loss":0.08922799694585837,"n_poly":1.32,"burn":41.53321036001948,"xr0":0.06,"xr_ov":0.02,"cov":0.05032318942436391,"cov_res":0.4177863300939433,"mis_res":0.25,"cd":0.65,"eta_v":0.85,"wall":0.000965438109878401,"hf":0.0002345835862811804,"ro_loss":0.98,"t_prim0":731.0371827564371,"t_prima":420,"k_int":2,"r_th":0.6,"k_tick":1.1587881555704223,"k_jet":0.005,"f_floor":15,"f_fw":400,"k_cab":0.35,"cab_q":1.5,"cab_model":1,"leak_floor":0.3,"leak_fw":0.00529734107537258,"k_floor":1.003419309652263,"k_ind":0.03265282579623439,"f_abs":8000,"k_win":0.3,"fric0":1.8479475605609046,"fric1":0.12,"fric2":0.018,"fric_cold":1.2,"a_idle":5.1887408207057674e-05,"fast_idle":1,"prime_rev":0.7443756334908427,"catch_x":2.4,"react":0.3,"k_st":1,"k_hiss":6.541746651658922,"pack":1.7434902998281796,"bay_exh":0.4409732071662665,"k_nl":0.7884990991797303,"dw_tau":0.8054969365610245,"gain":1,"k_st_mesh":0.42,"k_st_comm":0.16,"k_st_mot":0.14,"k_st_bb":0.9,"k_st_out":1,"k_st_cab":1.2,"k_st_bb2":0.48,"k_st_bbhi":0.22,"k_st_am":0.5,"k_st_am_far":0.55,"k_st_am_out":0.48,"k_st_am_cab":0.48,"st_lead":0.03,"rod":0.1651,"ev_d":0.038099999999999995,"ev_lift":0.01143,"adv_dur":282.0,"lsa":114.0,"in_dur":282.0,"in_lsa":114.0,"ex_dur":282.0,"ex_lsa":112.0,"x_len":0.0,"tail_d":0.044449999999999996,"v_ac":0.009163954184270904,"sn_d":0.1016,"sn_len":0.1016,"muf":[["tmm","triflow",0.21140420000000001,0.18455639999999998,0.11838939999999998,0.08255,0.19685,0.0508,0.044449999999999996,0.044449999999999996,0.1,0.0015,0.000762,0.11,650,0.08,0.0254],["pipe",3.07,0.044449999999999996,0.0],["tmm","perf_through",0.5715,0.13017499999999999,0.044449999999999996,0.2,3,0.22,0.45,0.33,0.0015,0.003175,550,0.06]],"tail_len":0.5,"cabin_len":2.19,"cabin_h":1.11,"r_ac":3900.0},"v13.1":{"q_comb":4.6,"heat_loss":0.08922799694585837,"n_poly":1.32,"burn":41.53321036001948,"xr0":0.06,"xr_ov":0.02,"cov":0.05032318942436391,"cov_res":0.4177863300939433,"mis_res":0.25,"cd":0.65,"eta_v":0.85,"wall":0.000965438109878401,"hf":0.0002345835862811804,"ro_loss":0.98,"t_prim0":731.0371827564371,"t_prima":420,"k_int":2,"r_th":0.6,"k_tick":1.1587881555704223,"k_jet":0.005,"f_floor":15,"f_fw":400,"k_cab":0.35,"cab_q":1.5,"cab_model":1,"leak_floor":0.3,"leak_fw":0.00529734107537258,"k_floor":1.003419309652263,"k_ind":0.03265282579623439,"f_abs":8000,"k_win":0.3,"fric0":1.8479475605609046,"fric1":0.12,"fric2":0.018,"fric_cold":1.2,"a_idle":5.1887408207057674e-05,"fast_idle":1,"prime_rev":0.7443756334908427,"catch_x":2.4,"react":0.3,"k_st":1,"k_hiss":6.541746651658922,"pack":1.7434902998281796,"bay_exh":0.4409732071662665,"k_nl":0.7884990991797303,"dw_tau":0.8054969365610245,"gain":1,"k_st_mesh":0.42,"k_st_comm":0.16,"k_st_mot":0.14,"k_st_bb":0.9,"k_st_out":1,"k_st_cab":1.2,"k_st_bb2":0.48,"k_st_bbhi":0.22,"k_st_am":0.5,"k_st_am_far":0.55,"k_st_am_out":0.48,"k_st_am_cab":0.48,"st_lead":0.03,"rod":0.1651,"ev_d":0.038099999999999995,"ev_lift":0.01143,"adv_dur":282.0,"lsa":114.0,"in_dur":282.0,"in_lsa":114.0,"ex_dur":282.0,"ex_lsa":112.0,"x_len":0.0,"tail_d":0.044449999999999996,"v_ac":0.009163954184270904,"sn_d":0.1016,"sn_len":0.1016,"muf":[["tmm","triflow",0.21140420000000001,0.18455639999999998,0.11838939999999998,0.08255,0.19685,0.0508,0.044449999999999996,0.044449999999999996,0.1,0.0015,0.000762,0.11,650,0.08,0.0254,1],["pipe",3.07,0.044449999999999996,0.0],["tmm","perf_through",0.5715,0.13017499999999999,0.044449999999999996,0.2,3,0.22,0.45,0.33,0.0015,0.003175,550,0.06,1]],"tail_len":0.5,"cabin_len":2.19,"cabin_h":1.11,"r_ac":3900.0}}},"p997":{"car":{"name":"2006 Porsche 911 Carrera 4S (997.1, M97.01 3.8)","bore":0.099,"stroke":0.0828,"rod":0.127,"cr":11.8,"fire_order":[1,6,2,4,3,5],"left_bank":[4,5,6],"adv_dur":252,"lsa":116,"ev_d":0.048648946545634474,"ev_lift":0.01,"camx":1,"lifters":"hydraulic","idle_rpm":670,"redline":7300,"inertia":0.16,"carb":"EFI, 1 electronic throttle body, two-stage resonance manifold","th_area_wot":0.004536459791783661,"v_man":0.0045,"v_ac":0.008,"sn_len":0.25,"sn_d":0.09,"th_len":0.1,"exhaust":"3-1","prim_len":0.3,"prim_x":[1,1.08,1.16],"prim_d":0.042,"head_len":0.25,"pipe_d":0.057,"x_len":0,"inter_len":0.6,"muf":[["chamber",0.3,4,0.15],["pipe",0.2,0.05,0],["chamber",0.22,10,0.5],["pipe",0.12,0.045,0],["chamber",0.22,10,0.5]],"tail_len":0.12,"bank_dl":0.04,"cross_mouth":true,"mouth":[[-0.02,0.52,0.3],[-0.02,-0.52,0.3]],"intake":[0.55,0,0.95],"covers":[[0.75,0.42,0.42],[0.75,-0.42,0.42]],"starter_pos":[1.3,0.22,0.35],"head":[2.15,0.35,1],"cabin_len":1.9,"cabin_h":1,"rear_engine":true,"eng_shadow":0.8,"st_ratio":54.888888888888886,"st_teeth":130,"st_free_rpm":16000,"st_i_free":70,"st_i_lock":850,"st_v_test":11.5,"st_v_lock":6,"r_batt":0.008,"st_kind":"bosch_pm","v_batt":12.5,"st_i_sat":10000,"spark0":10},"phys":{"q_comb":4.6,"heat_loss":0.06610091648405512,"n_poly":1.32,"burn":59.99578391306203,"xr0":0.05,"xr_ov":0.02,"cov":0.06304950659730628,"cov_res":0.07350737085237899,"mis_res":0.25,"cd":0.65,"eta_v":0.85,"wall":0.0004324461745975085,"hf":0.0007752814319838385,"ro_loss":0.98,"t_prim0":737.582808083124,"t_prima":420,"k_int":2,"r_th":0.6,"k_tick":0,"k_jet":0.36735134466175745,"f_floor":15,"f_fw":374.7941575801626,"k_cab":0.35,"cab_q":2.510320898642034,"cab_model":1,"leak_floor":0.3,"leak_fw":0.0035613790028847887,"k_floor":0.9663133290728505,"k_ind":0.5,"f_abs":8000,"k_win":0.3,"fric0":1.0197330664957027,"fric1":0.12,"fric2":0.018,"fric_cold":1.8,"a_idle":3.5e-05,"fast_idle":1,"prime_rev":2,"catch_x":2.4,"react":0.3,"k_st":1,"k_hiss":40,"pack":2.212846561320972,"bay_exh":0.17202969592712633,"k_nl":0.9301919446001485,"dw_tau":0.10000000000000002,"gain":1,"choke_tau":6,"k_st_mesh":0.32,"k_st_comm":0.16,"k_st_mot":0.14,"k_st_bb":1.7,"k_st_out":1.15,"k_st_cab":1,"k_st_bb2":1.25,"k_st_bbhi":0.38,"k_st_am":0.86,"k_st_am_far":0.88,"k_st_am_out":0.7,"k_st_am_cab":0.58,"st_lead":0.14},"sets":{"p1":{"q_comb":4.6,"heat_loss":0.06610091648405512,"n_poly":1.32,"burn":59.99578391306203,"xr0":0.05,"xr_ov":0.02,"cov":0.06304950659730628,"cov_res":0.07350737085237899,"mis_res":0.25,"cd":0.65,"eta_v":0.85,"wall":0.0004324461745975085,"hf":0.0007752814319838385,"ro_loss":0.98,"t_prim0":737.582808083124,"t_prima":420,"k_int":2,"r_th":0.6,"k_tick":295.0049331123548,"k_jet":0.49517551650628466,"f_floor":15,"f_fw":374.7941575801626,"k_cab":0.35,"cab_q":2.510320898642034,"cab_model":1,"leak_floor":0.3,"leak_fw":0.004429059907930195,"k_floor":0.9663133290728505,"k_ind":0.5,"f_abs":2586.2046359831593,"k_win":0.3,"fric0":1.0197330664957027,"fric1":0.12,"fric2":0.018,"fric_cold":1.8,"a_idle":3.5e-05,"fast_idle":1,"prime_rev":2,"catch_x":2.4,"react":0.3,"k_st":1,"k_hiss":16.19051453942309,"pack":2.325741107606418,"bay_exh":0.17202969592712633,"k_nl":0.9301919446001485,"dw_tau":0.10000000000000002,"gain":1,"choke_tau":6,"k_st_mesh":0.32,"k_st_comm":0.16,"k_st_mot":0.14,"k_st_bb":1.7,"k_st_out":1.15,"k_st_cab":1,"k_st_bb2":1.25,"k_st_bbhi":0.38,"k_st_am":0.86,"k_st_am_far":0.88,"k_st_am_out":0.7,"k_st_am_cab":0.58,"st_lead":0.14,"k_vf":0,"vf_n":2.2074897731823055,"tick_alpha":0,"vf_width":45,"vf_floor":0,"vf_rpm_ref":4500,"vf_rpm_exp":2.5,"k_psharp":2,"pack_r":0.2,"cat_early":1,"int_res":1,"int_kres":3e-07,"int_tube_l":0.3,"int_tube_d":0.045,"int_flap_lo":2600,"int_flap_hi":5100,"int_flap_thr":0.3,"cover_q1":2,"cover_q2":2,"vcam":1,"vcam_lift_lo":3.6,"vcam_lift_hi":11,"vcam_on_rpm":3000,"vcam_on_thr":0.28,"vcam_off_rpm":2500,"vcam_off_thr":0.18,"vcam_phase_rpm":4600,"vcam_range":40,"vcam_load0":0.35,"vcam_load1":0.65,"vcam_load_floor":0.15,"vcam_pull_rpm":6000,"vcam_pull_min":0.55,"vcam_pull_span":2500,"vcam_flap_lo":3000,"vcam_flap_hi":5500,"vcam_flap_thr":0.35,"vcam_flap_vol":0.55,"int_settle":0.25},"p2":{"q_comb":4.6,"heat_loss":0.06610091648405512,"n_poly":1.32,"burn":59.99578391306203,"xr0":0.05,"xr_ov":0.02,"cov":0.06304950659730628,"cov_res":0.07350737085237899,"mis_res":0.25,"cd":0.65,"eta_v":0.85,"wall":0.0004324461745975085,"hf":0.0007752814319838385,"ro_loss":0.98,"t_prim0":737.582808083124,"t_prima":420,"k_int":2,"r_th":0.6,"k_tick":295.0049331123548,"k_jet":0.49517551650628466,"f_floor":15,"f_fw":374.7941575801626,"k_cab":0.35,"cab_q":2.510320898642034,"cab_model":1,"leak_floor":0.3,"leak_fw":0.004429059907930195,"k_floor":0.9663133290728505,"k_ind":0.5,"f_abs":2586.2046359831593,"k_win":0.3,"fric0":1.0197330664957027,"fric1":0.12,"fric2":0.018,"fric_cold":1.8,"a_idle":3.5e-05,"fast_idle":1,"prime_rev":2,"catch_x":2.4,"react":0.3,"k_st":1,"k_hiss":16.19051453942309,"pack":2.325741107606418,"bay_exh":0.17202969592712633,"k_nl":0.9301919446001485,"dw_tau":0.10000000000000002,"gain":1,"choke_tau":6,"k_st_mesh":0.32,"k_st_comm":0.16,"k_st_mot":0.14,"k_st_bb":1.7,"k_st_out":1.15,"k_st_cab":1,"k_st_bb2":1.25,"k_st_bbhi":0.38,"k_st_am":0.86,"k_st_am_far":0.88,"k_st_am_out":0.7,"k_st_am_cab":0.58,"st_lead":0.14,"k_vf":0,"vf_n":2.2074897731823055,"tick_alpha":0,"vf_width":45,"vf_floor":0,"vf_rpm_ref":4500,"vf_rpm_exp":2.5,"k_psharp":2,"pack_r":0.2,"cat_early":1,"int_res":1,"int_kres":3e-07,"int_tube_l":0.3,"int_tube_d":0.045,"int_flap_lo":2600,"int_flap_hi":5100,"int_flap_thr":0.3,"cover_q1":2,"cover_q2":2,"vcam":1,"vcam_lift_lo":3.6,"vcam_lift_hi":11,"vcam_on_rpm":3000,"vcam_on_thr":0.28,"vcam_off_rpm":2500,"vcam_off_thr":0.18,"vcam_phase_rpm":4600,"vcam_range":40,"vcam_load0":0.35,"vcam_load1":0.65,"vcam_load_floor":0.15,"vcam_pull_rpm":6000,"vcam_pull_min":0.55,"vcam_pull_span":2500,"vcam_flap_lo":3000,"vcam_flap_hi":5500,"vcam_flap_thr":0.35,"vcam_flap_vol":0.55,"int_settle":0.25,"muf":[["chamber",0.3,4,0.15],["pipe",0.2,0.05,0],["perf",0.44,0.057,0.44,0.18,0.12,0.5],["pipe",0.12,0.045,0]]},"p3":{"q_comb":4.6,"heat_loss":0.06610091648405512,"n_poly":1.32,"burn":59.99578391306203,"xr0":0.05,"xr_ov":0.02,"cov":0.06304950659730628,"cov_res":0.07350737085237899,"mis_res":0.25,"cd":0.65,"eta_v":0.85,"wall":0.0004324461745975085,"hf":0.0007752814319838385,"ro_loss":0.98,"t_prim0":737.582808083124,"t_prima":420,"k_int":2,"r_th":0.6,"k_tick":295.0049331123548,"k_jet":0.49517551650628466,"f_floor":15,"f_fw":374.7941575801626,"k_cab":0.35,"cab_q":2.510320898642034,"cab_model":1,"leak_floor":0.3,"leak_fw":0.004429059907930195,"k_floor":0.9663133290728505,"k_ind":0.5,"f_abs":2586.2046359831593,"k_win":0.3,"fric0":1.0197330664957027,"fric1":0.12,"fric2":0.018,"fric_cold":1.8,"a_idle":3.5e-05,"fast_idle":1,"prime_rev":2,"catch_x":2.4,"react":0.3,"k_st":1,"k_hiss":16.19051453942309,"pack":2.325741107606418,"bay_exh":0.17202969592712633,"k_nl":0.9301919446001485,"dw_tau":0.10000000000000002,"gain":1,"choke_tau":6,"k_st_mesh":0.32,"k_st_comm":0.16,"k_st_mot":0.14,"k_st_bb":1.7,"k_st_out":1.15,"k_st_cab":1,"k_st_bb2":1.25,"k_st_bbhi":0.38,"k_st_am":0.86,"k_st_am_far":0.88,"k_st_am_out":0.7,"k_st_am_cab":0.58,"st_lead":0.14,"k_vf":0,"vf_n":2.2074897731823055,"tick_alpha":0,"vf_width":45,"vf_floor":0,"vf_rpm_ref":4500,"vf_rpm_exp":2.5,"k_psharp":2,"pack_r":0.2,"cat_early":1,"int_res":1,"int_kres":3e-07,"int_tube_l":0.3,"int_tube_d":0.045,"int_flap_lo":2600,"int_flap_hi":5100,"int_flap_thr":0.3,"cover_q1":2,"cover_q2":2,"vcam":1,"vcam_lift_lo":3.6,"vcam_lift_hi":11,"vcam_on_rpm":3000,"vcam_on_thr":0.28,"vcam_off_rpm":2500,"vcam_off_thr":0.18,"vcam_phase_rpm":4600,"vcam_range":40,"vcam_load0":0.35,"vcam_load1":0.65,"vcam_load_floor":0.15,"vcam_pull_rpm":6000,"vcam_pull_min":0.55,"vcam_pull_span":2500,"vcam_flap_lo":3000,"vcam_flap_hi":5500,"vcam_flap_thr":0.35,"vcam_flap_vol":0.55,"int_settle":0.25,"muf":[["chamber",0.3,4,0.15],["pipe",0.2,0.05,0],["chamber",0.12,11,0.1],["par",[[0.2,0.052,0],[0.0049,0.032744007085266764,0],[0.2,0.018,0]],0.16,11,0.1],["par",[[0.2,0.052,0],[0.0049,0.032744007085266764,0]],0.12,11,0.5],["pipe",0.1,0.058,0]]},"p4":{"q_comb":4.6,"heat_loss":0.06610091648405512,"n_poly":1.32,"burn":59.99578391306203,"xr0":0.05,"xr_ov":0.02,"cov":0.06304950659730628,"cov_res":0.07350737085237899,"mis_res":0.25,"cd":0.65,"eta_v":0.85,"wall":0.0004324461745975085,"hf":0.0007752814319838385,"ro_loss":0.98,"t_prim0":737.582808083124,"t_prima":420,"k_int":2,"r_th":0.6,"k_tick":295.0049331123548,"k_jet":0.49517551650628466,"f_floor":15,"f_fw":374.7941575801626,"k_cab":0.35,"cab_q":2.510320898642034,"cab_model":1,"leak_floor":0.3,"leak_fw":0.004429059907930195,"k_floor":0.9663133290728505,"k_ind":0.5,"f_abs":2586.2046359831593,"k_win":0.3,"fric0":1.0197330664957027,"fric1":0.12,"fric2":0.018,"fric_cold":1.8,"a_idle":3.5e-05,"fast_idle":1,"prime_rev":2,"catch_x":2.4,"react":0.3,"k_st":1,"k_hiss":16.19051453942309,"pack":2.325741107606418,"bay_exh":0.17202969592712633,"k_nl":0.9301919446001485,"dw_tau":0.10000000000000002,"gain":1,"choke_tau":6,"k_st_mesh":0.32,"k_st_comm":0.16,"k_st_mot":0.14,"k_st_bb":1.7,"k_st_out":1.15,"k_st_cab":1,"k_st_bb2":1.25,"k_st_bbhi":0.38,"k_st_am":0.86,"k_st_am_far":0.88,"k_st_am_out":0.7,"k_st_am_cab":0.58,"st_lead":0.14,"k_vf":0,"vf_n":2.2074897731823055,"tick_alpha":0,"vf_width":45,"vf_floor":0,"vf_rpm_ref":4500,"vf_rpm_exp":2.5,"k_psharp":2,"pack_r":0.2,"cat_early":1,"int_res":1,"int_kres":3e-07,"int_tube_l":0.3,"int_tube_d":0.045,"int_flap_lo":2600,"int_flap_hi":5100,"int_flap_thr":0.3,"cover_q1":2,"cover_q2":2,"vcam":1,"vcam_lift_lo":3.6,"vcam_lift_hi":11,"vcam_on_rpm":3000,"vcam_on_thr":0.28,"vcam_off_rpm":2500,"vcam_off_thr":0.18,"vcam_phase_rpm":4600,"vcam_range":40,"vcam_load0":0.35,"vcam_load1":0.65,"vcam_load_floor":0.15,"vcam_pull_rpm":6000,"vcam_pull_min":0.55,"vcam_pull_span":2500,"vcam_flap_lo":3000,"vcam_flap_hi":5500,"vcam_flap_thr":0.35,"vcam_flap_vol":0.55,"int_settle":0.25,"muf":[["chamber",0.3,4,0.15],["pipe",0.2,0.05,0],["pipe",0.16,0.062,0],["cavity",0.12,0.24183051916579923,0,0.005,0.24183051916579923,0.001,0],["cavity",0.06,0.24183051916579923,0,0.0049,0.07647352483049281,0.001,0],["cavity",0.105,0.24183051916579923,0,0.0049,0.07647352483049281,0.001,0,0.025,0.5],["pipe",0.16,0.062,0],["pipe",0.25,0.057,0]]},"p5":{"q_comb":4.6,"heat_loss":0.06610091648405512,"n_poly":1.32,"burn":59.99578391306203,"xr0":0.05,"xr_ov":0.02,"cov":0.06304950659730628,"cov_res":0.07350737085237899,"mis_res":0.25,"cd":0.65,"eta_v":0.85,"wall":0.0004324461745975085,"hf":0.0007752814319838385,"ro_loss":0.98,"t_prim0":737.582808083124,"t_prima":420,"k_int":2,"r_th":0.6,"k_tick":295.0049331123548,"k_jet":0.49517551650628466,"f_floor":15,"f_fw":374.7941575801626,"k_cab":0.35,"cab_q":2.510320898642034,"cab_model":1,"leak_floor":0.3,"leak_fw":0.004429059907930195,"k_floor":0.9663133290728505,"k_ind":0.5,"f_abs":2586.2046359831593,"k_win":0.3,"fric0":1.0197330664957027,"fric1":0.12,"fric2":0.018,"fric_cold":1.8,"a_idle":3.5e-05,"fast_idle":1,"prime_rev":2,"catch_x":2.4,"react":0.3,"k_st":1,"k_hiss":16.19051453942309,"pack":2.325741107606418,"bay_exh":0.17202969592712633,"k_nl":0.9301919446001485,"dw_tau":0.10000000000000002,"gain":1,"choke_tau":6,"k_st_mesh":0.32,"k_st_comm":0.16,"k_st_mot":0.14,"k_st_bb":1.7,"k_st_out":1.15,"k_st_cab":1,"k_st_bb2":1.25,"k_st_bbhi":0.38,"k_st_am":0.86,"k_st_am_far":0.88,"k_st_am_out":0.7,"k_st_am_cab":0.58,"st_lead":0.14,"k_vf":0,"vf_n":2.2074897731823055,"tick_alpha":0,"vf_width":45,"vf_floor":0,"vf_rpm_ref":4500,"vf_rpm_exp":2.5,"k_psharp":2,"pack_r":0.2,"cat_early":1,"int_res":1,"int_kres":3e-07,"int_tube_l":0.3,"int_tube_d":0.045,"int_flap_lo":2600,"int_flap_hi":5100,"int_flap_thr":0.3,"cover_q1":2,"cover_q2":2,"vcam":1,"vcam_lift_lo":3.6,"vcam_lift_hi":11,"vcam_on_rpm":3000,"vcam_on_thr":0.28,"vcam_off_rpm":2500,"vcam_off_thr":0.18,"vcam_phase_rpm":4600,"vcam_range":40,"vcam_load0":0.35,"vcam_load1":0.65,"vcam_load_floor":0.15,"vcam_pull_rpm":6000,"vcam_pull_min":0.55,"vcam_pull_span":2500,"vcam_flap_lo":3000,"vcam_flap_hi":5500,"vcam_flap_thr":0.35,"vcam_flap_vol":0.55,"int_settle":0.25,"muf":[["chamber",0.3,4,0.15],["pipe",0.2,0.05,0],["pipe",0.2,0.057,0],["par",[[0.0049,0.06752777206453653,0]],0.1344,18,0.5],["par",[[0.3,0.057,0]],0.1216,18,0],["par",[[0.0049,0.07647352483049281,0]],0.064,18,0],["pipe",0.3,0.057,0],["pipe",0.25,0.057,0]]},"p6":{"q_comb":4.6,"heat_loss":0.06610091648405512,"n_poly":1.32,"burn":59.99578391306203,"xr0":0.05,"xr_ov":0.02,"cov":0.06304950659730628,"cov_res":0.07350737085237899,"mis_res":0.25,"cd":0.65,"eta_v":0.85,"wall":0.0004324461745975085,"hf":0.0007752814319838385,"ro_loss":0.98,"t_prim0":737.582808083124,"t_prima":420,"k_int":2,"r_th":0.6,"k_tick":295.0049331123548,"k_jet":0.49517551650628466,"f_floor":15,"f_fw":374.7941575801626,"k_cab":0.35,"cab_q":2.510320898642034,"cab_model":1,"leak_floor":0.3,"leak_fw":0.004429059907930195,"k_floor":0.9663133290728505,"k_ind":0.5,"f_abs":2586.2046359831593,"k_win":0.3,"fric0":1.0197330664957027,"fric1":0.12,"fric2":0.018,"fric_cold":1.8,"a_idle":3.5e-05,"fast_idle":1,"prime_rev":2,"catch_x":2.4,"react":0.3,"k_st":1,"k_hiss":16.19051453942309,"pack":2.325741107606418,"bay_exh":0.17202969592712633,"k_nl":0.9301919446001485,"dw_tau":0.10000000000000002,"gain":1,"choke_tau":6,"k_st_mesh":0.32,"k_st_comm":0.16,"k_st_mot":0.14,"k_st_bb":1.7,"k_st_out":1.15,"k_st_cab":1,"k_st_bb2":1.25,"k_st_bbhi":0.38,"k_st_am":0.86,"k_st_am_far":0.88,"k_st_am_out":0.7,"k_st_am_cab":0.58,"st_lead":0.14,"k_vf":0,"vf_n":2.2074897731823055,"tick_alpha":0,"vf_width":45,"vf_floor":0,"vf_rpm_ref":4500,"vf_rpm_exp":2.5,"k_psharp":2,"pack_r":0.2,"cat_early":1,"int_res":1,"int_kres":3e-07,"int_tube_l":0.3,"int_tube_d":0.045,"int_flap_lo":2600,"int_flap_hi":5100,"int_flap_thr":0.3,"cover_q1":2,"cover_q2":2,"vcam":1,"vcam_lift_lo":3.6,"vcam_lift_hi":11,"vcam_on_rpm":3000,"vcam_on_thr":0.28,"vcam_off_rpm":2500,"vcam_off_thr":0.18,"vcam_phase_rpm":4600,"vcam_range":40,"vcam_load0":0.35,"vcam_load1":0.65,"vcam_load_floor":0.15,"vcam_pull_rpm":6000,"vcam_pull_min":0.55,"vcam_pull_span":2500,"vcam_flap_lo":3000,"vcam_flap_hi":5500,"vcam_flap_thr":0.35,"vcam_flap_vol":0.55,"int_settle":0.25,"muf":[["chamber",0.3,4,0.15],["pipe",0.32,0.05,0],["pipe",0.16,0.062,0],["cavity",0.12,0.24183051916579923,0,0.005,0.24183051916579923,0.001,0],["cavity",0.06,0.24183051916579923,0,0.0049,0.07647352483049281,0.001,0],["cavity",0.105,0.24183051916579923,0,0.0049,0.07647352483049281,0.001,0,0.025,0.5],["pipe",0.16,0.062,0],["pipe",0.25,0.057,0]]},"p7":{"q_comb":4.6,"heat_loss":0.06610091648405512,"n_poly":1.32,"burn":59.99578391306203,"xr0":0.05,"xr_ov":0.02,"cov":0.06304950659730628,"cov_res":0.07350737085237899,"mis_res":0.25,"cd":0.65,"eta_v":0.85,"wall":0.0004324461745975085,"hf":0.0007752814319838385,"ro_loss":0.98,"t_prim0":737.582808083124,"t_prima":420,"k_int":2,"r_th":0.6,"k_tick":295.0049331123548,"k_jet":0.49517551650628466,"f_floor":15,"f_fw":374.7941575801626,"k_cab":0.35,"cab_q":2.510320898642034,"cab_model":1,"leak_floor":0.3,"leak_fw":0.004429059907930195,"k_floor":0.9663133290728505,"k_ind":0.5,"f_abs":2586.2046359831593,"k_win":0.3,"fric0":1.0197330664957027,"fric1":0.12,"fric2":0.018,"fric_cold":1.8,"a_idle":3.5e-05,"fast_idle":1,"prime_rev":2,"catch_x":2.4,"react":0.3,"k_st":1,"k_hiss":16.19051453942309,"pack":2.325741107606418,"bay_exh":0.17202969592712633,"k_nl":0.9301919446001485,"dw_tau":0.10000000000000002,"gain":1,"choke_tau":6,"k_st_mesh":0.32,"k_st_comm":0.16,"k_st_mot":0.14,"k_st_bb":1.7,"k_st_out":1.15,"k_st_cab":1,"k_st_bb2":1.25,"k_st_bbhi":0.38,"k_st_am":0.86,"k_st_am_far":0.88,"k_st_am_out":0.7,"k_st_am_cab":0.58,"st_lead":0.14,"k_vf":0,"vf_n":2.2074897731823055,"tick_alpha":0,"vf_width":45,"vf_floor":0,"vf_rpm_ref":4500,"vf_rpm_exp":2.5,"k_psharp":2,"pack_r":0.2,"cat_early":1,"int_res":1,"int_kres":3e-07,"int_tube_l":0.3,"int_tube_d":0.045,"int_flap_lo":2600,"int_flap_hi":5100,"int_flap_thr":0.3,"cover_q1":2,"cover_q2":2,"vcam":1,"vcam_lift_lo":3.6,"vcam_lift_hi":11,"vcam_on_rpm":3000,"vcam_on_thr":0.28,"vcam_off_rpm":2500,"vcam_off_thr":0.18,"vcam_phase_rpm":4600,"vcam_range":40,"vcam_load0":0.35,"vcam_load1":0.65,"vcam_load_floor":0.15,"vcam_pull_rpm":6000,"vcam_pull_min":0.55,"vcam_pull_span":2500,"vcam_flap_lo":3000,"vcam_flap_hi":5500,"vcam_flap_thr":0.35,"vcam_flap_vol":0.55,"int_settle":0.25,"muf":[["chamber",0.3,4,0.15],["pipe",0.2,0.05,0],["pipe",0.16,0.062,0.2],["cavity",0.12,0.24183051916579923,0.3,0.005,0.24183051916579923,0.001,0],["cavity",0.06,0.24183051916579923,0,0.0049,0.07647352483049281,0.001,0],["cavity",0.105,0.24183051916579923,0,0.0049,0.07647352483049281,0.001,0,0.025,0.5],["pipe",0.16,0.062,0],["pipe",0.25,0.057,0]]}}}}/*END_V8P_DATA*/;
    // v13.2: same v13.1 geometry with rev=2 (wider-band fit + fs/2 TMM filters). Generic param only.
    if (DATA.cadillac59 && DATA.cadillac59.sets && DATA.cadillac59.sets['v13.1'] && !DATA.cadillac59.sets['v13.2']) {
      DATA.cadillac59.sets['v13.2'] = JSON.parse(JSON.stringify(DATA.cadillac59.sets['v13.1']));
      var _m132 = DATA.cadillac59.sets['v13.2'].muf, _i132;
      for (_i132 = 0; _m132 && _i132 < _m132.length; _i132++) {
        if (_m132[_i132] && _m132[_i132][0] === 'tmm') _m132[_i132][_m132[_i132].length - 1] = 2;
      }
    }

    // v13.3: same v13.1 geometry with rev=3 (rev-2 fit, continuous fs/4 output interpolation, decimation lag taken off the bulk delay). Generic param only.
    if (DATA.cadillac59 && DATA.cadillac59.sets && DATA.cadillac59.sets['v13.1'] && !DATA.cadillac59.sets['v13.3']) {
      DATA.cadillac59.sets['v13.3'] = JSON.parse(JSON.stringify(DATA.cadillac59.sets['v13.1']));
      var _m133 = DATA.cadillac59.sets['v13.3'].muf, _i133;
      for (_i133 = 0; _m133 && _i133 < _m133.length; _i133++) {
        if (_m133[_i133] && _m133[_i133][0] === 'tmm') _m133[_i133][_m133[_i133].length - 1] = 3;
      }
    }

    // v13.4: v13.3 (v13.1 geometry, TMM rev=3) + gas_dyn=1 (cycle-resolved exhaust gas temperature). Generic param only.
    if (DATA.cadillac59 && DATA.cadillac59.sets && DATA.cadillac59.sets['v13.3'] && !DATA.cadillac59.sets['v13.4']) {
      DATA.cadillac59.sets['v13.4'] = JSON.parse(JSON.stringify(DATA.cadillac59.sets['v13.3']));
      DATA.cadillac59.sets['v13.4'].gas_dyn = 1;
    }

    // v14: v13.4 with flow noise removed from the model (k_jet = 0: no tailpipe jet noise; k_vf = 0: no exhaust-valve flow noise).
    // Both were about 80 dB under the radiated exhaust sound. Generic params only; every older set stays bit-identical.
    if (DATA.cadillac59 && DATA.cadillac59.sets && DATA.cadillac59.sets['v13.4'] && !DATA.cadillac59.sets['v14']) {
      DATA.cadillac59.sets['v14'] = JSON.parse(JSON.stringify(DATA.cadillac59.sets['v13.4']));
      DATA.cadillac59.sets['v14'].k_jet = 0;
      DATA.cadillac59.sets['v14'].k_vf = 0;
    }

    // v14.1: v14 with the can shells as 3-D modal chambers (generic kinds 'triflow3d' / 'perf_through3d'; same gas
    // paths, same 1-D tubes, same perforate formula). Changed numbers, each from a source, none fitted to recordings:
    //  - triflow louvre porosity 0.10 -> 0.11: US2182945 Fig. 3 shows about 20 louvres round the 1 3/4 in tube
    //    (14 on the drawn 250 deg arc), opening 0.030 in (US1931736): 20 x 0.030 / (pi x 1.75) = 0.109.
    //  - gas temperature and Mach number from the engine model at warm idle (tools/can_gas.cjs): p_man 0.285 atm,
    //    520 rpm, T_prim 798.6 K; first can fed by the group-1 pipe at T_AMB + 0.65 (T_prim - T_AMB) = 621.7 K, the
    //    resonator at T_tail = 519.6 K; volume flow eta_v p_man/p_atm V_d rpm/120 T/T_AMB per bank over the tube area:
    //    M = 0.0073 (muffler, 2 in inlet) and 0.0087 (resonator, 1 3/4 in). Replaces the estimates 650 K / 0.08, 550 K / 0.06.
    //  - TMM rev 4: exact causal impulse response from a dense grid split into the same hybrid FIR + IIR filter (compileCan4).
    if (DATA.cadillac59 && DATA.cadillac59.sets && DATA.cadillac59.sets['v14'] && !DATA.cadillac59.sets['v14.1']) {
      DATA.cadillac59.sets['v14.1'] = JSON.parse(JSON.stringify(DATA.cadillac59.sets['v14']));
      var _m141 = DATA.cadillac59.sets['v14.1'].muf, _i141;
      for (_i141 = 0; _m141 && _i141 < _m141.length; _i141++) {
        var _c141 = _m141[_i141];
        if (!_c141 || _c141[0] !== 'tmm') continue;
        if (_c141[1] === 'triflow') { _c141[1] = 'triflow3d'; _c141[10] = 0.11; _c141[14] = 621.7; _c141[15] = 0.0073; _c141[17] = 4; }
        else if (_c141[1] === 'perf_through') { var _n141 = _c141[6] | 0; _c141[1] = 'perf_through3d'; _c141[9 + _n141] = 519.6; _c141[10 + _n141] = 0.0087; _c141[11 + _n141] = 4; }
      }
    }

    // v15: v14.1 plus a second, prebuilt gas state for every TMM can (generic param muf_cruise, same geometry): the engine
    // model at steady cruise, 0.4 x redline (2000 rpm) and quarter throttle (tools/can_gas.cjs cadillac59 v14.1 cruise):
    // p_man 0.607 atm, T_prim 922.1 K; muffler 701.9 K, M 0.0634; resonator 576.1 K, M 0.0750. The idle state is v14.1's.
    // At run time each can uses the design state nearest the engine model's live mean-flow Mach number (log midpoint,
    // +-20 % hysteresis) with a short warm-up and crossfade on a change. The TMM rev-5 perforate fix applies to the 1-D can
    // kinds only; v14.1/v15 cans are 3-D (rev 4), which never used the 1-D perforate cascade.
    if (DATA.cadillac59 && DATA.cadillac59.sets && DATA.cadillac59.sets['v14.1'] && !DATA.cadillac59.sets['v15']) {
      DATA.cadillac59.sets['v15'] = JSON.parse(JSON.stringify(DATA.cadillac59.sets['v14.1']));
      var _m15 = DATA.cadillac59.sets['v15'].muf, _mc15 = [], _i15;
      for (_i15 = 0; _m15 && _i15 < _m15.length; _i15++) {
        var _c15 = _m15[_i15];
        if (!_c15 || _c15[0] !== 'tmm') { _mc15.push(null); continue; }
        var _k15 = _c15.slice();
        if (_k15[1] === 'triflow3d') { _k15[14] = 701.9; _k15[15] = 0.0634; }
        else if (_k15[1] === 'perf_through3d') { var _n15 = _k15[6] | 0; _k15[9 + _n15] = 576.1; _k15[10 + _n15] = 0.0750; }
        _mc15.push(_k15);
      }
      DATA.cadillac59.sets['v15'].muf_cruise = _mc15;
    }

    // p8 (v12 family): v14.1 parameters with the four cans back on the plain v12 1-D muffler list (no TMM, no 3-D shell).
    // Owner-authorised small structured search (29 candidates, 1-3 chambers per can, packing 0-1, internal tube 0.1/0.3 m)
    // scored on the 142 dry windows; the simplest layout won in every leave-one-source-out fold. Per bank:
    //  - muffler (Walker 21158): one chamber, 20.25 in shell (catalog), area of the 3.25 x 7.75 in oval over the 2 in pipe
    //    = 6.296875 (catalog), packing 0 (searched; TriFlow is reactive, US3841435 / US3498406 mention no packing);
    //  - connector 3.07 m of 1 3/4 in pipe (p5 sourced lengths, as v13-v15);
    //  - resonator (Walker 21159): one chamber, 22.5 in shell, 5 1/8 in round over the 2 in pipe = 6.56640625 (Walker
    //    catalog spec), packing 0 (searched; Direct Flow, no packing listed). Chamber count per can: searched.
    // Generic 'chamber'/'pipe' items only; no Cadillac-only code. Every older set stays bit-identical.
    if (DATA.cadillac59 && DATA.cadillac59.sets && DATA.cadillac59.sets['v14.1'] && !DATA.cadillac59.sets['p8']) {
      DATA.cadillac59.sets['p8'] = JSON.parse(JSON.stringify(DATA.cadillac59.sets['v14.1']));
      DATA.cadillac59.sets['p8'].muf = [['chamber', 0.51435, 6.296875, 0], ['pipe', 3.07, 0.04445, 0], ['chamber', 0.5715, 6.56640625, 0]];
    }

    // p9 (v12 family): p8 plus a realistic amount of damping, set from physics and sources, not from the scorecard.
    // NACA Report 1192 (Davis et al. 1954): plain expansion chambers without flow match lossless plane-wave theory,
    // so reactive cans damp little at rest; Sullivan (JASA 66, 1979): louvred/perforated cans with flow "can be very
    // dissipative" through the flow-induced resistance of the perforation. Per can: segmented concentric perforated-
    // duct model (Sullivan & Crocker 1978; Munjal 2014 ch. 4) of the real shell and 1 3/4 in tube, Rao & Munjal (1986)
    // grazing-flow resistance 7.337e-3 (1 + 72.23 M)/sigma plus orifice visco-thermal resistance, Kirchhoff wall
    // losses at the gas temperature (Sutherland viscosity), gas state midway between the engine model's warm idle and
    // cruise (v14.1/v15 can_gas). Its dissipative attenuation (0.4-2.6 dB, rising with frequency) is mapped onto the
    // existing generic chamber terms by least squares in octave bands: chamber pack 0.25 (muffler) / 0.13
    // (resonator), with the set's pack 1.743 unchanged and pack_r 0.30 (broadband part). Generic params only.
    if (DATA.cadillac59 && DATA.cadillac59.sets && DATA.cadillac59.sets['p8'] && !DATA.cadillac59.sets['p9']) {
      DATA.cadillac59.sets['p9'] = JSON.parse(JSON.stringify(DATA.cadillac59.sets['p8']));
      DATA.cadillac59.sets['p9'].muf = [['chamber', 0.51435, 6.296875, 0.25], ['pipe', 3.07, 0.04445, 0], ['chamber', 0.5715, 6.56640625, 0.13]];
      DATA.cadillac59.sets['p9'].pack_r = 0.30;
    }

    // p10 (v12 family): p9 with damping at the upper end of the physically plausible range (same method as p9, not a
    // score fit): engine-model cruise gas state (muffler 701.9 K, M 0.0634; resonator 576.1 K, M 0.0750) instead of the
    // idle/cruise midpoint; Rao & Munjal resistance x1.25 (scatter of the empirical fit, estimate); muffler louvre
    // porosity 0.08 (14 louvres round the tube, US2182945 Fig. 3, lower bound) plus Sullivan (1979) cross-flow
    // resistance 0.514 d M/(l sigma) because all TriFlow flow crosses the louvres; resonator porosity 0.15 with 1.5 mm
    // holes (estimate). Dissipative attenuation 1.0-6.0 dB (muffler) and 0.7-3.4 dB (resonator) per octave band,
    // mapped as in p9: chamber pack 0.39 / 0.24, set pack 1.743 unchanged, pack_r 0.31. No packing material.
    if (DATA.cadillac59 && DATA.cadillac59.sets && DATA.cadillac59.sets['p9'] && !DATA.cadillac59.sets['p10']) {
      DATA.cadillac59.sets['p10'] = JSON.parse(JSON.stringify(DATA.cadillac59.sets['p9']));
      DATA.cadillac59.sets['p10'].muf = [['chamber', 0.51435, 6.296875, 0.39], ['pipe', 3.07, 0.04445, 0], ['chamber', 0.5715, 6.56640625, 0.24]];
      DATA.cadillac59.sets['p10'].pack_r = 0.31;
    }

    // Sound v16 (Cadillac): v15 with its four TMM cans replaced by plain long single chambers in the sourced Walker shells,
    // following the inverse-cans analysis (scratch/inverse_cans: the recordings want an even can loss 125 Hz-2 kHz; v15's
    // cans act as a strong low-pass). Per bank: muffler 21158, 20.25 in shell, 3.25 x 7.75 in oval over the 2 in pipe =
    // area ratio 6.296875 (Walker catalog); connector 3.07 m of 1 3/4 in pipe (p5 sourced); resonator 21159, 22.5 in shell,
    // 5 1/8 in round = 6.56640625 (Walker catalog); tail 0.5 m (p5). No extended tubes, no reverse-flow internals, no
    // Helmholtz branch (none sourced). Damping: the p9 physics (damp.py: Rao & Munjal 1986 louvre/perforate resistance,
    // Kirchhoff wall loss, idle/cruise midpoint gas), chamber pack 0.25 / 0.13, pack_r 0.30. New generic key muf_t1: the
    // first can's gas is at the group-1 temperature, as for v15's first TMM can (can_gas.cjs). Generic params only.
    if (DATA.cadillac59 && DATA.cadillac59.sets && DATA.cadillac59.sets['v15'] && !DATA.cadillac59.sets['v16']) {
      DATA.cadillac59.sets['v16'] = JSON.parse(JSON.stringify(DATA.cadillac59.sets['v15']));
      delete DATA.cadillac59.sets['v16'].muf_cruise;
      DATA.cadillac59.sets['v16'].muf = [['chamber', 0.51435, 6.296875, 0.25], ['pipe', 3.07, 0.04445, 0], ['chamber', 0.5715, 6.56640625, 0.13]];
      DATA.cadillac59.sets['v16'].pack_r = 0.30;
      DATA.cadillac59.sets['v16'].muf_t1 = 1;
    }

    // Sound v17 (Cadillac): v16 plus two generic physical effects, no fitted values.
    // (1) port_bc: every exhaust area step >= 1.2 is a resistive junction with the mean-flow Borda-Carnot loss
    //     Dp = K rho U u (Ronneberger 1967; Munjal 2014 ch. 2; expansion K = (1 - A1/A2)^2, contraction 0.5 (1 - A2/A1), Idelchik).
    // (2) t_axial: the exhaust gas cools along each bank from the head pipe to the tail (steady 1-D heat balance, Incropera
    //     & DeWitt ch. 8; inside h from Dittus-Boelter 1930; flow from the engine model). Outside coefficient h_o = 20 W/m2K
    //     for a still car: natural convection (Churchill & Chu 1975, 8-9 W/m2K on the 1 3/4 in pipe) plus radiation from
    //     oxidised steel (emissivity 0.79, Incropera Table A.11; 8-12 W/m2K at 420-500 K) = 16-21 W/m2K. Driving would be
    //     higher (Hilpert cross-flow, ~50-80 W/m2K); 20 is the conservative still-air value.
    //     At warm idle the resonator runs at ~387 K and the tail at ~380 K instead of 505 K, so the tail half-wave and the
    //     two cans' pass bands no longer coincide. Generic params only.
    if (DATA.cadillac59 && DATA.cadillac59.sets && DATA.cadillac59.sets['v16'] && !DATA.cadillac59.sets['v17']) {
      DATA.cadillac59.sets['v17'] = JSON.parse(JSON.stringify(DATA.cadillac59.sets['v16']));
      DATA.cadillac59.sets['v17'].port_bc = 1;
      DATA.cadillac59.sets['v17'].t_axial = 20;
    }

    // Sound 'v19' (Cadillac, picker-only): v17 plus generic outside-path propagation and the road-noise
    // block. Sourced: Levine & Schwinger 1948 / Ando 1968 directivity; ISO 9613-1 air absorption (20 C, 70 % RH: estimate);
    // Delany & Bazley 1970 ground with Nord2000 class G (dense asphalt / concrete) sigma = 20 000 kPa s/m2 (Plovsing 2006;
    // surface of the recordings unknown: estimate); mic-geometry coherence: height sd 0.3 m, distance sd 0.5 m (ESTIMATES,
    // hand-held video camera). Noise: CNOSSOS-EU 2015/996 cat. 1; 8.20-15 bias-ply tyre, section ~208 mm, rim 15 in,
    // overall ~0.72 m -> cavity mean diameter ~0.55 m (tyre size tables: estimate); 1959 wheelbase 130 in = 3.30 m (AMA),
    // rear axle ~1.47 m ahead of the bumper (estimate); interior: mass law 9 kg/m2 floor (0.9 mm steel + trim: estimate),
    // cavity peak +6 dB re 250 Hz (estimate), wind 65 dB(A) at 100 km/h with v^5.5 (George 1990 range 5-6; anchor estimate).
    if (DATA.cadillac59 && DATA.cadillac59.sets && DATA.cadillac59.sets['v17'] && !DATA.cadillac59.sets['v19']) {
      DATA.cadillac59.sets['v19'] = JSON.parse(JSON.stringify(DATA.cadillac59.sets['v17']));
      DATA.cadillac59.sets['v19'].out_path = { dir: 1, air: 1, ground: 20000, coh_h: 0.3, coh_d: 0.5, T: 20, RH: 70 };
      DATA.cadillac59.sets['v19'].noise = { tyre_w: 208, d_mean: 0.55, wheelbase: 3.30, axle_x: 1.37, floor_m: 9, r_near: 1.0, cav_q: 8, cav_db: 6, wind_db100: 65, wind_n: 5.5, shape: 1 };
    }

    // Sound 'v20' (Cadillac, picker-only): v19 with the Walker 21159 resonator
    // modelled as a straight-through perforated can (generic kind perf_through3d; same sourced shell,
    // gas path and idle gas state as v14.1 / v15), not as a plain expansion chamber. The 21158 muffler
    // stays the v16 long chamber with p9 pack. Motive: two similar-length plain chambers plus the 0.5 m
    // tail share a pass near c/(2L) ≈ 400 Hz (NACA 1192 / Munjal expansion-chamber TL is zero at kL=π).
    // That pass dominates closed-throttle cabin tilt. Walker 21159 is a Direct-Flow (straight-through)
    // resonator, which fills the plane-wave pass. Parameters only; older sets stay bit-identical.
    if (DATA.cadillac59 && DATA.cadillac59.sets && DATA.cadillac59.sets['v19'] && !DATA.cadillac59.sets['v20']) {
      DATA.cadillac59.sets['v20'] = JSON.parse(JSON.stringify(DATA.cadillac59.sets['v19']));
      var _m20 = DATA.cadillac59.sets['v20'].muf, _i20, _res20 = null;
      if (DATA.cadillac59.sets['v15'] && DATA.cadillac59.sets['v15'].muf) {
        for (_i20 = 0; _i20 < DATA.cadillac59.sets['v15'].muf.length; _i20++) {
          if (DATA.cadillac59.sets['v15'].muf[_i20] && DATA.cadillac59.sets['v15'].muf[_i20][0] === 'tmm' && String(DATA.cadillac59.sets['v15'].muf[_i20][1]).indexOf('perf_through') === 0) {
            _res20 = JSON.parse(JSON.stringify(DATA.cadillac59.sets['v15'].muf[_i20]));
            break;
          }
        }
      }
      if (_m20 && _res20) {
        for (_i20 = 0; _i20 < _m20.length; _i20++) {
          if (_m20[_i20] && _m20[_i20][0] === 'chamber' && _i20 > 0) { _m20[_i20] = _res20; break; }
        }
      }
    }

    var P_ATM = 101325.0, R_GAS = 287.0, G_EX = 1.30, G_AIR = 1.40, T_AMB = 293.0, C_AIR = 343.0, RHO_AIR = 1.2;
    /* Generic transfer-matrix muffler elements (v13).
     * Frequency-domain 2x2 TMM from standard elements (tube, area change, Helmholtz
     * shunt, perforated duct with Lee–Ih / Sullivan perforate impedance, including a
     * louvred-tube equivalent), precomputed at load time and converted to a short
     * causal IIR for transmission and input reflection. Parameters only; no car-specific
     * code. This file is included inside the v8p module (and is node-testable on its own).
     */
    function createEngineTmm() {
      'use strict';
      var PI = Math.PI, MU_AIR = 1.8e-5; // Pa·s, estimate: air/exhaust dynamic viscosity (weakly T-dependent)

      function cx(r, i) { return { r: r, i: i || 0 }; }
      function cadd(a, b) { return cx(a.r + b.r, a.i + b.i); }
      function csub(a, b) { return cx(a.r - b.r, a.i - b.i); }
      function cmul(a, b) { return cx(a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r); }
      function cdiv(a, b) {
        var d = b.r * b.r + b.i * b.i;
        if (d < 1e-300) return cx(0, 0);
        return cx((a.r * b.r + a.i * b.i) / d, (a.i * b.r - a.r * b.i) / d);
      }
      function cabs(a) { return Math.sqrt(a.r * a.r + a.i * a.i); }
      function cneg(a) { return cx(-a.r, -a.i); }
      function cscale(a, s) { return cx(a.r * s, a.i * s); }
      function cexpj(th) { return cx(Math.cos(th), Math.sin(th)); }
      function cinv(a) { return cdiv(cx(1, 0), a); }
      function ccosh(z) { // cosh(x+iy) = cosh x cos y + i sinh x sin y
        var x = z.r, y = z.i;
        return cx(Math.cosh(x) * Math.cos(y), Math.sinh(x) * Math.sin(y));
      }
      function csinh(z) {
        var x = z.r, y = z.i;
        return cx(Math.sinh(x) * Math.cos(y), Math.cosh(x) * Math.sin(y));
      }

      function tId() { var o = cx(1, 0), z = cx(0, 0); return [[o, z], [z, o]]; }
      function tMul(A, B) {
        return [
          [cadd(cmul(A[0][0], B[0][0]), cmul(A[0][1], B[1][0])), cadd(cmul(A[0][0], B[0][1]), cmul(A[0][1], B[1][1]))],
          [cadd(cmul(A[1][0], B[0][0]), cmul(A[1][1], B[1][0])), cadd(cmul(A[1][0], B[0][1]), cmul(A[1][1], B[1][1]))]
        ];
      }
      function tChain(list) {
        var T = tId(), i;
        for (i = 0; i < list.length; i++) T = tMul(T, list[i]);
        return T;
      }

      // Straight tube, p–U convention. Visco-thermal loss + mean-flow (simple 1/(1-M²) wavenumber).
      function tTube(L, area, omega, rho, c0, M) {
        if (L < 0) L = 0;
        if (area < 1e-10) area = 1e-10;
        if (omega < 1e-6) omega = 1e-6;
        var P = 2 * Math.sqrt(PI * area); // perimeter
        // Kirchhoff/Kirchhoff-style wall attenuation (Munjal): α ≈ P/S * sqrt(ω ν) / (2 c) * const
        var nu = MU_AIR / rho, alpha = (P / (2 * area)) * Math.sqrt(nu * omega) / c0;
        if (alpha > 5) alpha = 5;
        if (!M) M = 0;
        if (M > 0.4) M = 0.4;
        if (M < -0.4) M = -0.4;
        var k = omega / c0, den = 1 - M * M;
        if (den < 0.5) den = 0.5;
        var g = cx(alpha, k / den);
        var gL = cscale(g, L);
        var ch = ccosh(gL), sh = csinh(gL);
        var Zc = rho * c0 / area;
        // convective: slightly different Zc±; keep a single Zc (standard first-order).
        return [[ch, cscale(sh, Zc)], [cscale(sh, 1 / Zc), ch]];
      }

      // Point-side-branch / Helmholtz shunt at a junction: T = [[1,0],[1/Zh,1]]
      function tShunt(Zh) {
        return [[cx(1, 0), cx(0, 0)], [cinv(Zh), cx(1, 0)]];
      }
      function zHelmholtz(omega, rho, c0, V, Lneq, Aneck, r) {
        // Zh = r + j ω ρ Lneq / A + ρ c² / (j ω V)
        var jw = cx(0, omega);
        var inert = cscale(jw, rho * Lneq / Math.max(Aneck, 1e-10));
        var comp = cdiv(cx(rho * c0 * c0 / Math.max(V, 1e-10), 0), jw);
        return cadd(cx(r || 0, 0), cadd(inert, comp));
      }

      // Lee–Ih / Sullivan-style specific perforate impedance, then /porosity.
      // inclLouver: treat orifices as louvred slots by using dHole as the opening width
      // and a slightly higher end correction (tangential louvers, US2182945).
      function zPerforate(omega, rho, c0, sigma, tWall, dHole, M, inclLouver) {
        if (omega < 1e-6) omega = 1e-6;
        if (sigma < 1e-5) sigma = 1e-5;
        if (sigma > 0.9) sigma = 0.9;
        if (dHole < 1e-5) dHole = 1e-5;
        var k = omega / c0;
        // viscous + thermal resistance (Kirchhoff) and grazing-flow term (Lee & Ih / Ingard–Dean style)
        var rv = Math.sqrt(8 * MU_AIR * rho * omega) * (1 + tWall / dHole) / (rho * c0);
        var rg = 0.3 * Math.abs(M || 0); // grazing Mach resistance, standard coefficient ~0.3
        var R = (rv + rg) / sigma;
        // mass end correction: 0.85 d (1-0.7√σ) plus wall thickness; louvers get a slightly longer equivalent
        var corr = 0.85 * dHole * (1 - 0.7 * Math.sqrt(sigma));
        if (inclLouver) corr *= 1.2;
        var X = k * (tWall + corr) / sigma;
        return cx(R, X); // Z/(ρc)
      }

      // One segment of a concentric (or multi-duct + cavity) perforated duct, lumped:
      // inner tube Δx, shunt to a cavity volume ΔV through perforate, cavity closed axially in the slice
      // (the cavity is also advanced as an equivalent tube when coupleCavityAxial is true).
      // For a single concentric resonator we cascade these and close the cavity at both ends.

      // 4x4 complex helpers for two coupled ducts (inner + annulus).
      function m4id() {
        var I = cx(1, 0), Z = cx(0, 0), r = [], i, j;
        for (i = 0; i < 4; i++) { r[i] = []; for (j = 0; j < 4; j++) r[i][j] = i === j ? I : Z; }
        return r;
      }
      function m4mul(A, B) {
        var C = [], i, j, k, s;
        for (i = 0; i < 4; i++) {
          C[i] = [];
          for (j = 0; j < 4; j++) {
            s = cx(0, 0);
            for (k = 0; k < 4; k++) s = cadd(s, cmul(A[i][k], B[k][j]));
            C[i][j] = s;
          }
        }
        return C;
      }
      // 2-port tube as a 4x4 acting on one duct (0=inner or 2=annulus), the other duct unchanged.
      function m4tube(T, which) {
        var M = m4id(), b = which ? 2 : 0;
        M[b][b] = T[0][0]; M[b][b + 1] = T[0][1];
        M[b + 1][b] = T[1][0]; M[b + 1][b + 1] = T[1][1];
        return M;
      }
      // Perforate coupling over a slice: U1' = U1 - (p1-p2)/Zlink, U2' = U2 + (p1-p2)/Zlink; p unchanged.
      // State is [p1,U1,p2,U2] on the right in terms of the left (thin layer, no length).
      function m4link(Zlink) {
        var M = m4id(), y = cinv(Zlink); // y = 1/Zlink
        // U1' = U1 - y p1 + y p2
        M[1][0] = cneg(y); M[1][2] = y; // M[1][1] already 1
        // U2' = U2 + y p1 - y p2
        M[3][0] = y; M[3][2] = cneg(y);
        return M;
      }
      function t2inv(Q) {
        // invert 2x2 [[a,b],[c,d]]
        var det = csub(cmul(Q[0][0], Q[1][1]), cmul(Q[0][1], Q[1][0]));
        return [[cdiv(Q[1][1], det), cdiv(cneg(Q[0][1]), det)], [cdiv(cneg(Q[1][0]), det), cdiv(Q[0][0], det)]];
      }

      // Proper Sullivan / Munjal concentric perforated duct: inner + annular waveguides
      // coupled by a distributed perforate, annulus closed at both ends, reduced to a 2x2 on the inner tube.
      // At high frequency the perforate mass closes the holes and the inner tube passes (textbook behaviour).
      function tPerforatedConcentricLegacy(L, dTube, dShell, sigma, tWall, dHole, omega, rho, c0, M, nseg, inclLouver) {
        if (nseg < 1) nseg = 6;
        var dx = L / nseg, a1 = PI * dTube * dTube / 4, a2 = PI * dShell * dShell / 4 - a1;
        if (a2 < 1e-6) a2 = 1e-6;
        var Zpc = zPerforate(omega, rho, c0, sigma, tWall, dHole, M, inclLouver);
        var Zohm = cscale(Zpc, rho * c0);
        var Awall = PI * dTube * dx;
        var Zlink = cdiv(Zohm, cx(Math.max(Awall, 1e-9), 0));
        var M = m4id(), s, half = dx * 0.5;
        var T1 = tTube(half, a1, omega, rho, c0, M), T2 = tTube(half, a2, omega, rho, c0, 0);
        var Lk = m4link(Zlink);
        var seg = m4mul(m4mul(m4mul(m4tube(T1, 0), m4tube(T2, 1)), Lk), m4mul(m4tube(T1, 0), m4tube(T2, 1)));
        for (s = 0; s < nseg; s++) M = m4mul(seg, M); // X_right = M X_left, state [p1,U1,p2,U2]
        // Close the annulus: U2L = 0, U2R = 0. X_R = M X_L.
        // U2R = M[3][0] p1L + M[3][1] U1L + M[3][2] p2L = 0  (U2L=0)
        var m30 = M[3][0], m31 = M[3][1], m32 = M[3][2];
        // p2L = -(m30 p1L + m31 U1L) / m32
        var inv = cinv(m32);
        if (cabs(m32) < 1e-18) inv = cx(0, 0);
        var k0 = cneg(cmul(inv, m30)), k1 = cneg(cmul(inv, m31));
        // p1R = M00 p1L + M01 U1L + M02 p2L
        // U1R = M10 p1L + M11 U1L + M12 p2L
        var Q00 = cadd(M[0][0], cmul(M[0][2], k0));
        var Q01 = cadd(M[0][1], cmul(M[0][2], k1));
        var Q10 = cadd(M[1][0], cmul(M[1][2], k0));
        var Q11 = cadd(M[1][1], cmul(M[1][2], k1));
        // X_R = Q X_L ; standard TMM is X_L = T X_R, so T = Q^{-1}
        var T = t2inv([[Q00, Q01], [Q10, Q11]]);
        if (!T || !isFinite(T[0][0].r)) return tTube(L, a1, omega, rho, c0, M);
        return T;
      }

      function tPerforatedConcentricFixed(L, dTube, dShell, sigma, tWall, dHole, omega, rho, c0, Mach, nseg, inclLouver) {
        if (nseg < 1) nseg = 6;
        var dx = L / nseg, a1 = PI * dTube * dTube / 4, a2 = PI * dShell * dShell / 4 - a1;
        if (a2 < 1e-6) a2 = 1e-6;
        var Zpc = zPerforate(omega, rho, c0, sigma, tWall, dHole, Mach, inclLouver);
        var Zohm = cscale(Zpc, rho * c0);
        var Awall = PI * dTube * dx;
        var Zlink = cdiv(Zohm, cx(Math.max(Awall, 1e-9), 0));
        var MM = m4id(), s, half = dx * 0.5;
        var T1 = tTube(half, a1, omega, rho, c0, Mach), T2 = tTube(half, a2, omega, rho, c0, 0);
        var Lk = m4link(Zlink);
        var seg = m4mul(m4mul(m4mul(m4tube(T1, 0), m4tube(T2, 1)), Lk), m4mul(m4tube(T1, 0), m4tube(T2, 1)));
        for (s = 0; s < nseg; s++) MM = m4mul(seg, MM);
        var m30 = MM[3][0], m31 = MM[3][1], m32 = MM[3][2];
        var inv = cinv(m32);
        if (cabs(m32) < 1e-18) inv = cx(0, 0);
        var k0 = cneg(cmul(inv, m30)), k1 = cneg(cmul(inv, m31));
        var Q00 = cadd(MM[0][0], cmul(MM[0][2], k0));
        var Q01 = cadd(MM[0][1], cmul(MM[0][2], k1));
        var Q10 = cadd(MM[1][0], cmul(MM[1][2], k0));
        var Q11 = cadd(MM[1][1], cmul(MM[1][2], k1));
        var TT = t2inv([[Q00, Q01], [Q10, Q11]]);
        if (!TT || !isFinite(TT[0][0].r)) return tTube(L, a1, omega, rho, c0, Mach);
        return TT;
      }
      // v15 (TMM rev >= 5): causal concentric perforated duct. The rev 1-4 cascade above composes the half-tube matrices
      // (which map the right-hand state to the left-hand state, X_L = T X_R) with a perforate link written the other way
      // round (X_R from X_L), then inverts the result. That flips the link's sign (a perforate with negative admittance)
      // and the travel direction of the waves: refined, it does not tend to the expansion-chamber limit, and its impulse
      // response has energy before t = 0. Rev 5 builds the same segmented coupled-duct model (inner tube and annulus as 1-D
      // waveguides, perforate as a distributed series impedance between them, annulus closed at both ends: Sullivan &
      // Crocker, JASA 64 (1978) 207-215; Munjal, Acoustics of Ducts and Mufflers, 2nd ed. 2014, ch. 4) with each element
      // in its own, consistent direction. Segments: at least nseg and no longer than 1.5 cm (midpoint rule, as the 3-D
      // model). Kept apart so rev <= 4 is unchanged.
      function tPerforatedConcentricCausal(L, dTube, dShell, sigma, tWall, dHole, omega, rho, c0, Mach, nseg, inclLouver) {
        // Solved as a nodal network (the generic tOfNet below), not as a 4x4 matrix power: with a well-coupled perforate the
        // difference mode decays fast along the duct and a 4x4 cascade overflows; the nodal solve stays well conditioned.
        var ns = Math.max(nseg | 0, Math.ceil(L / 0.015), 1);
        var dx = L / ns, a1 = PI * dTube * dTube / 4, a2 = PI * dShell * dShell / 4 - a1, k;
        if (a2 < 1e-6) a2 = 1e-6;
        var Zohm = cscale(zPerforate(omega, rho, c0, sigma, tWall, dHole, Mach, inclLouver), rho * c0);
        var Zlink = cdiv(Zohm, cx(Math.max(PI * dTube * dx, 1e-9), 0));
        var Th = tTube(dx * 0.5, a1, omega, rho, c0, Mach), Tf = tTube(dx, a1, omega, rho, c0, Mach);
        var Ah = tTube(dx * 0.5, a2, omega, rho, c0, 0), Af = tTube(dx, a2, omega, rho, c0, 0);
        var link = [[cx(1, 0), Zlink], [cx(0, 0), cx(1, 0)]];
        // nodes: 0 = inlet; 1..ns inner mid-segment nodes; ns+1..2ns annulus mid nodes; 2ns+1, 2ns+2 closed annulus ends; last = outlet
        var elems = [], N = 2 * ns + 4, out = N - 1, aL = 2 * ns + 1, aR = 2 * ns + 2;
        elems.push({ i: 0, j: 1, T: Th });
        for (k = 1; k < ns; k++) elems.push({ i: k, j: k + 1, T: Tf });
        elems.push({ i: ns, j: out, T: Th });
        elems.push({ i: aL, j: ns + 1, T: Ah });
        for (k = 1; k < ns; k++) elems.push({ i: ns + k, j: ns + k + 1, T: Af });
        elems.push({ i: 2 * ns, j: aR, T: Ah });
        for (k = 1; k <= ns; k++) elems.push({ i: k, j: ns + k, T: link });
        var TT = tOfNet(N, elems, [], omega, rho, c0);
        if (!TT || !isFinite(TT[0][0].r)) return tTube(L, a1, omega, rho, c0, Mach);
        return TT;
      }
      function tPerforatedConcentric(L, dTube, dShell, sigma, tWall, dHole, omega, rho, c0, Mach, nseg, inclLouver, rev) {
        if (rev >= 5) return tPerforatedConcentricCausal(L, dTube, dShell, sigma, tWall, dHole, omega, rho, c0, Mach, nseg, inclLouver);
        if (rev >= 1) return tPerforatedConcentricFixed(L, dTube, dShell, sigma, tWall, dHole, omega, rho, c0, Mach, nseg, inclLouver);
        return tPerforatedConcentricLegacy(L, dTube, dShell, sigma, tWall, dHole, omega, rho, c0, Mach, nseg, inclLouver);
      }

      // Solve 2x2 T for S-params given port areas (characteristic Zc = ρc/A).
      // S is defined on travelling pressure waves p± = (p ± Zc U)/2  with U in the +x sense at port 1
      // and +x (out of the can) at port 2, i.e. U2 leaving the can.
      function tToS(T, area1, area2, rho, c0, rev) {
        var Z1 = rho * c0 / area1, Z2 = rho * c0 / area2;
        var A = T[0][0], B = T[0][1], C = T[1][0], D = T[1][1];
        // p1 = A p2 + B U2, U1 = C p2 + D U2
        // p1+ = (p1 + Z1 U1)/2, p1- = (p1 - Z1 U1)/2
        // p2+ (leaving can) = (p2 + Z2 U2)/2, p2- (entering can from downstream) = (p2 - Z2 U2)/2
        // Express p1-, p2+ in terms of p1+, p2-.
        // Den = A Z2 + B + Z1 C Z2 + Z1 D
        var Den = cadd(cadd(cscale(A, Z2), B), cadd(cscale(C, Z1 * Z2), cscale(D, Z1)));
        // S11 = (A Z2 + B - Z1 C Z2 - Z1 D) / Den
        var N11 = csub(cadd(cscale(A, Z2), B), cadd(cscale(C, Z1 * Z2), cscale(D, Z1)));
        // S21 = 2 Z2 / Den   (pressure-wave transmission; power has a Z factor)
        // More carefully: p2+ / p1+ at p2-=0.
        // When p2-=0, p2 = Z2 U2, so p1 = (A Z2 + B) U2, U1 = (C Z2 + D) U2
        // p1+ = (p1+Z1 U1)/2 = (A Z2+B + Z1 C Z2 + Z1 D) U2 / 2 = Den U2 / 2
        // p2+ = (p2+Z2 U2)/2 = Z2 U2
        // S21 = p2+ / p1+ = 2 Z2 / Den
        var N21 = cx(2 * Z2, 0);
        // S12 = 2 Z1 / Den  (reciprocal if T is reciprocal and Z1=Z2; general: 2 Z1 / Den)
        var N12 = cx(2 * Z1, 0);
        // S22 = (-A Z2 + B - Z1 C Z2 + Z1 D) / Den ? Let me derive p2-=1, p1+=0...
        // Standard formulas:
        // S22 = (-A Z2 + B + Z1 C Z2 - Z1 D) / Den  I'll use a two-case numerical approach for safety.
        var S11 = cdiv(N11, Den);
        var S21 = cdiv(N21, Den);
        var S12 = cdiv(N12, Den);
        // Case: drive from port 2. Set p2-=1, p1+=0 and solve.
        // I use the closed form (Munjal / standard, Z-normalized slightly differently).
        // N22 = -A Z2 + B + Z1 C Z2 - Z1 D
        var N22;
        if (rev >= 1) {
          // p2+/p2- at p1+=0 with U both +x: (-A Z2 + B - Z1 C Z2 + Z1 D)/Den
          N22 = cadd(csub(B, cscale(A, Z2)), csub(cscale(D, Z1), cscale(C, Z1 * Z2)));
        } else {
          // v13 legacy (wrong Z1 terms; kept so released v13 stays bit-identical)
          N22 = cadd(csub(B, cscale(A, Z2)), csub(cscale(C, Z1 * Z2), cscale(D, Z1)));
        }
        var S22 = cdiv(N22, Den);
        return { S11: S11, S12: S12, S21: S21, S22: S22, Den: Den };
      }

      function tlOfS(S, area1, area2) {
        // Power TL (anechoic termination): -10 log( (A2/A1) |S21|^2 / something )
        // For pressure-wave S as defined above, transmitted power / incident = (area1/area2) * |S21|^2 * (Z2/Z1)? 
        // Zc ~ 1/area, p+ power ~ |p+|^2 area / (ρc).
        // Πinc = |p1+|^2 * area1 / (ρc), Πtr = |p2+|^2 * area2 / (ρc)
        // τ = (area2/area1) |S21|^2, TL = -10 log τ
        var tau = (area2 / area1) * (S.S21.r * S.S21.r + S.S21.i * S.S21.i);
        if (tau < 1e-16) tau = 1e-16;
        return -10 * Math.log(tau) / Math.LN10;
      }

      // ---- can builders (generic; params only) ---------------------------------
      // triflow: three-tube reverse-flow can with louvred tubes in the middle chamber
      //   and end-chamber resonator (US2182945 topology).
      // perf_through: straight-through perforated tube with N chambers (US3648803 / Sullivan).
      function tOfCan(kind, p, omega, rho, c0) {
        if (kind === 'perf_through') return tOfPerfThrough(p, omega, rho, c0);
        if (kind === 'perf_through3d') return tOfPerfThrough3d(p, omega, rho, c0);
        if (kind === 'triflow3d') return tOfTriflow3d(p, omega, rho, c0);
        return tOfTriflow(p, omega, rho, c0);
      }

      function tOfPerfThrough(p, omega, rho, c0) {
        // p: L, dShell, dTube, chFrac[] (sum 1), sigma, tWall, dHole, M
        var L = p.L, dS = p.dShell, dT = p.dTube, fr = p.chFrac || [1];
        var sigma = p.sigma, tw = p.tWall, dh = p.dHole, M = p.M || 0;
        var aT = PI * dT * dT / 4, i, T = tId();
        for (i = 0; i < fr.length; i++) {
          var Li = L * fr[i];
          if (Li < 1e-4) continue;
          T = tMul(T, tPerforatedConcentric(Li, dT, dS, sigma, tw, dh, omega, rho, c0, M, 8, false, p.rev));
        }
        return T;
      }

      function tOfTriflow(p, omega, rho, c0) {
        // p: L1,L2,L3, dIn, dRet, dOut, aOval, bOval (full widths), sigma, tWall, dHole, Lxover, M
        var L1 = p.L1, L2 = p.L2, L3 = p.L3;
        var aIn = PI * p.dIn * p.dIn / 4, aRet = PI * p.dRet * p.dRet / 4, aOut = PI * p.dOut * p.dOut / 4;
        var aShell = PI * (p.aOval * 0.5) * (p.bOval * 0.5); // ellipse area π a b with a,b = semi-axes = half widths
        var M = p.M || 0, sigma = p.sigma, tw = p.tWall, dh = p.dHole, Lx = p.Lxover || (0.05);
        // Path (US2182945, opposite-end inlet/outlet):
        //  inlet tube through ch15 (solid L1) + louvred L2 in ch14
        //  rear crossover + ch13 Helmholtz
        //  return louvred L2
        //  front crossover
        //  outlet: louvred in ch15 (L1) + louvred in ch14 (L2) + through ch13 (L3)
        //
        // Reduced 1-D TMM along the retroverted gas path, with the end and middle
        // chambers as Helmholtz / concentric perforate shunts (standard elements).
        var V15 = Math.max(1e-6, (aShell - aIn - aOut) * L1);
        var V14 = Math.max(1e-6, (aShell - aIn - aRet - aOut) * L2);
        var V13 = Math.max(1e-6, (aShell - aIn - aOut) * L3);
        // Neck of the rear resonator: opening in the rear crossover wall (US2182945).
        // Size is not stated; estimate a 1 in hole, neck length = wall + end correction.
        var dNeck = p.dNeck || 0.0254, Lneck = (tw || 0.0015) + 0.85 * dNeck;
        var Aneck = PI * dNeck * dNeck / 4;
        var r13 = p.rNeck || (0.5 * rho * c0 / Aneck * 0.05);
        var ZinH = zHelmholtz(omega, rho, c0, V13, Lneck, Aneck, r13);

        var parts = [];
        // inlet through end chamber 15 (solid)
        parts.push(tTube(L1, aIn, omega, rho, c0, M));
        // louvred inlet in middle chamber (concentric-style, cavity V14 assigned to this segment as the dominant coupling)
        // Use an equivalent circular shell whose area is V14/L2 so the concentric model sees the right cavity.
        var dEq14 = 2 * Math.sqrt((V14 / L2 + aIn) / PI);
        parts.push(tPerforatedConcentric(L2, p.dIn, dEq14, sigma, tw, dh, omega, rho, c0, M, 6, true, p.rev));
        // rear crossover + Helmholtz shunt (chamber 13)
        parts.push(tTube(Lx, aRet, omega, rho, c0, M));
        parts.push(tShunt(ZinH));
        // return tube, louvred, same middle cavity (the cavity was already used on the inlet pass;
        // a second concentric on the return captures the second pass; slightly double-counts cavity
        // compliance — we split V14 in half by using a smaller equivalent shell on the return).
        var dEq14b = 2 * Math.sqrt((0.5 * V14 / L2 + aRet) / PI);
        parts.push(tPerforatedConcentric(L2, p.dRet, dEq14b, sigma, tw, dh, omega, rho, c0, M, 6, true, p.rev));
        // front crossover
        parts.push(tTube(Lx, aOut, omega, rho, c0, M));
        // outlet through ch15 (louvred into V15 — spit / end resonator, US2182945)
        var dEq15 = 2 * Math.sqrt((V15 / L1 + aOut) / PI);
        parts.push(tPerforatedConcentric(L1, p.dOut, dEq15, sigma, tw, dh, omega, rho, c0, M, 4, true, p.rev));
        // outlet through middle (already partly accounted) and rear (solid through ch13)
        parts.push(tTube(L2 * 0.25, aOut, omega, rho, c0, M)); // residual outlet length in ch14 not already in the return path
        parts.push(tTube(L3, aOut, omega, rho, c0, M));
        return tChain(parts);
      }

      // ---- v14.1: 3-D modal chamber model (generic) ----------------------------------------------
      // Each can shell chamber is a rigid-walled box (equal-area rectangle with the real section aspect ratio:
      // oval -> a x b scaled by sqrt(pi/4); round -> square of the same area), axial length = chamber length.
      // Pressure from volume velocity: modal Green's function (Morse & Ingard 1968, ch. 9; Venkatesham, Tiwari &
      // Munjal, J. Sound Vib. 323 (2009) 1032-1044): transverse modes cos(m pi y/Ly) cos(n pi z/Lz), the axial
      // direction in closed form per transverse mode, g = -cos(kx x<) cos(kx (L - x>)) / (kx sin(kx L)),
      // kx^2 = kc^2 - kmn^2 (cut-on and evanescent modes alike; g is even in kx). Patches (perforated tube
      // segments, necks, open ends) are averaged over x and over their equal-area square. Wall loss: the
      // same Kirchhoff boundary-layer attenuation as tTube, on the chamber's own perimeter / area.
      // Internal tubes stay 1-D (tTube); perforate impedance is the same zPerforate. The can two-port follows
      // from nodal analysis of tubes + perforates + chamber impedance matrices.
      function cmplxCos(ar, ai, o) { o[0] = Math.cos(ar) * Math.cosh(ai); o[1] = -Math.sin(ar) * Math.sinh(ai); }
      function cmplxSin(ar, ai, o) { o[0] = Math.sin(ar) * Math.cosh(ai); o[1] = Math.cos(ar) * Math.sinh(ai); }
      function sincR(u) { return Math.abs(u) < 1e-9 ? 1 : Math.sin(u) / u; }
      // chamber: { L, Ly, Lz }, patches: [{ x0, x1, y, z, s }] -> K x K complex (re, im) of p_k / Q_j [Pa s/m^3]
      function chamberZ3d(ch, patches, omega, rho, c0) {
        var K = patches.length, re = new Float64Array(K * K), im = new Float64Array(K * K);
        var L = ch.L, Ly = ch.Ly, Lz = ch.Lz, k = omega / c0, i, j, m, n;
        var nu = MU_AIR / rho, alpha = ((2 * (Ly + Lz)) / (2 * Ly * Lz)) * Math.sqrt(nu * omega) / c0;
        if (alpha > 5) alpha = 5;
        // kc = k - j alpha ; kc^2 = k^2 - alpha^2 - 2 j k alpha
        var kc2r = k * k - alpha * alpha, kc2i = -2 * k * alpha;
        var sMin = 1e9; for (i = 0; i < K; i++) if (patches[i].s < sMin) sMin = patches[i].s;
        var mMax = Math.min(64, Math.ceil(6 * Ly / sMin)), nMax = Math.min(64, Math.ceil(6 * Lz / sMin));
        if (ch.modeCap !== undefined) { mMax = Math.min(mMax, ch.modeCap); nMax = Math.min(nMax, ch.modeCap); }   // validation only
        var phi = new Float64Array(K), Cr = new Float64Array(K), Ci = new Float64Array(K), Dr = new Float64Array(K), Di = new Float64Array(K);
        var t = [0, 0], t2 = [0, 0], t3 = [0, 0], t4 = [0, 0];
        var same = []; // identical x-interval flags
        for (i = 0; i < K; i++) { same[i] = []; for (j = 0; j < K; j++) same[i][j] = Math.abs(patches[i].x0 - patches[j].x0) < 1e-9 && Math.abs(patches[i].x1 - patches[j].x1) < 1e-9; }
        for (m = 0; m <= mMax; m++) {
          for (n = 0; n <= nMax; n++) {
            var km2 = (m * PI / Ly) * (m * PI / Ly) + (n * PI / Lz) * (n * PI / Lz);
            var eps = (m ? 2 : 1) * (n ? 2 : 1), lam = Ly * Lz / eps;
            var any = false;
            for (i = 0; i < K; i++) {
              var P = patches[i];
              phi[i] = Math.cos(m * PI * P.y / Ly) * sincR(m * PI * P.s / (2 * Ly)) * Math.cos(n * PI * P.z / Lz) * sincR(n * PI * P.s / (2 * Lz));
              if (Math.abs(phi[i]) > 1e-12) any = true;
            }
            if (!any) continue;
            // kx = sqrt(kc^2 - km^2) (principal); g is even in kx so the branch does not matter
            var ar = kc2r - km2, ai = kc2i, mod = Math.sqrt(Math.sqrt(ar * ar + ai * ai)), ang = 0.5 * Math.atan2(ai, ar);
            var kr = mod * Math.cos(ang), ki = mod * Math.sin(ang);
            // den = kx sin(kx L)
            cmplxSin(kr * L, ki * L, t);
            var denr = kr * t[0] - ki * t[1], deni = kr * t[1] + ki * t[0];
            var dd = denr * denr + deni * deni; if (dd < 1e-300) dd = 1e-300;
            // 1/den
            var ivr = denr / dd, ivi = -deni / dd;
            // per-patch x-averages: C = <cos kx x>, D = <cos kx (L-x)>
            for (i = 0; i < K; i++) {
              var a = patches[i].x0, b = patches[i].x1, h = b - a;
              if (h < 1e-9) {
                cmplxCos(kr * a, ki * a, t); Cr[i] = t[0]; Ci[i] = t[1];
                cmplxCos(kr * (L - a), ki * (L - a), t); Dr[i] = t[0]; Di[i] = t[1];
              } else {
                // (sin kb - sin ka) / (k h)
                cmplxSin(kr * b, ki * b, t); cmplxSin(kr * a, ki * a, t2);
                var nr = t[0] - t2[0], ni = t[1] - t2[1], qr = kr * h, qi = ki * h, qd = qr * qr + qi * qi;
                Cr[i] = (nr * qr + ni * qi) / qd; Ci[i] = (ni * qr - nr * qi) / qd;
                cmplxSin(kr * (L - a), ki * (L - a), t); cmplxSin(kr * (L - b), ki * (L - b), t2);
                nr = t[0] - t2[0]; ni = t[1] - t2[1];
                Dr[i] = (nr * qr + ni * qi) / qd; Di[i] = (ni * qr - nr * qi) / qd;
              }
            }
            for (i = 0; i < K; i++) {
              if (phi[i] === 0) continue;
              for (j = i; j < K; j++) {
                if (phi[j] === 0) continue;
                var w = phi[i] * phi[j] / lam, gr, gi;
                var Pi = patches[i], Pj = patches[j];
                if (same[i][j] && Pi.x1 - Pi.x0 > 1e-9) {
                  // E = (2/(kx h^2)) [ 1/2 (h sin kxL - (cos kx(2b-L) - cos kx(2a-L)) / (2 kx)) - sin(kx a) (sin kx(L-a) - sin kx(L-b)) / kx ]
                  var a2 = Pi.x0, b2 = Pi.x1, h2 = b2 - a2;
                  cmplxSin(kr * L, ki * L, t);                                   // sin kxL
                  cmplxCos(kr * (2 * b2 - L), ki * (2 * b2 - L), t2); cmplxCos(kr * (2 * a2 - L), ki * (2 * a2 - L), t3);
                  var c1r = t2[0] - t3[0], c1i = t2[1] - t3[1];                  // cos diff
                  // c1 / (2 kx)
                  var kd = kr * kr + ki * ki, x1r = (c1r * kr + c1i * ki) / (2 * kd), x1i = (c1i * kr - c1r * ki) / (2 * kd);
                  var term1r = 0.5 * (h2 * t[0] - x1r), term1i = 0.5 * (h2 * t[1] - x1i);
                  cmplxSin(kr * a2, ki * a2, t4);
                  cmplxSin(kr * (L - a2), ki * (L - a2), t2); cmplxSin(kr * (L - b2), ki * (L - b2), t3);
                  var s2r = t2[0] - t3[0], s2i = t2[1] - t3[1];
                  var pr = t4[0] * s2r - t4[1] * s2i, pim = t4[0] * s2i + t4[1] * s2r;   // sin(ka) * sdiff
                  var y2r = (pr * kr + pim * ki) / kd, y2i = (pim * kr - pr * ki) / kd; // / kx
                  var er = term1r - y2r, ei = term1i - y2i;
                  // * 2/(kx h^2)
                  var fr = 2 * kr / (kd * h2 * h2), fi = -2 * ki / (kd * h2 * h2);
                  var Er = er * fr - ei * fi, Ei = er * fi + ei * fr;
                  gr = -(Er * ivr - Ei * ivi); gi = -(Er * ivi + Ei * ivr);
                } else {
                  // left patch uses C, right patch uses D
                  var lft = (Pi.x0 + Pi.x1) <= (Pj.x0 + Pj.x1) ? i : j, rgt = lft === i ? j : i;
                  var pr2 = Cr[lft] * Dr[rgt] - Ci[lft] * Di[rgt], pi2 = Cr[lft] * Di[rgt] + Ci[lft] * Dr[rgt];
                  gr = -(pr2 * ivr - pi2 * ivi); gi = -(pr2 * ivi + pi2 * ivr);
                }
                re[i * K + j] += w * gr; im[i * K + j] += w * gi;
              }
            }
          }
        }
        // Z = j omega rho * sum ; symmetric fill
        for (i = 0; i < K; i++) for (j = i; j < K; j++) {
          var zr = -omega * rho * im[i * K + j], zi = omega * rho * re[i * K + j];
          re[i * K + j] = zr; im[i * K + j] = zi; re[j * K + i] = zr; im[j * K + i] = zi;
        }
        return { re: re, im: im, K: K };
      }
      // complex dense solve (partial pivoting): A (n x n, re/im) X = B (n x nr) in place, returns X re/im
      function csolve(n, Ar, Ai, nr, Br, Bi) {
        var i, j, k, p, r;
        for (k = 0; k < n; k++) {
          p = k; var best = Ar[k * n + k] * Ar[k * n + k] + Ai[k * n + k] * Ai[k * n + k];
          for (i = k + 1; i < n; i++) { var v = Ar[i * n + k] * Ar[i * n + k] + Ai[i * n + k] * Ai[i * n + k]; if (v > best) { best = v; p = i; } }
          if (best < 1e-300) return false;
          if (p !== k) {
            for (j = 0; j < n; j++) { var tr = Ar[k * n + j], ti = Ai[k * n + j]; Ar[k * n + j] = Ar[p * n + j]; Ai[k * n + j] = Ai[p * n + j]; Ar[p * n + j] = tr; Ai[p * n + j] = ti; }
            for (j = 0; j < nr; j++) { var ur = Br[k * nr + j], ui = Bi[k * nr + j]; Br[k * nr + j] = Br[p * nr + j]; Bi[k * nr + j] = Bi[p * nr + j]; Br[p * nr + j] = ur; Bi[p * nr + j] = ui; }
          }
          var dr = Ar[k * n + k], di = Ai[k * n + k], dd = dr * dr + di * di, ir = dr / dd, ii = -di / dd;
          for (i = k + 1; i < n; i++) {
            var fr = Ar[i * n + k] * ir - Ai[i * n + k] * ii, fi = Ar[i * n + k] * ii + Ai[i * n + k] * ir;
            if (fr === 0 && fi === 0) continue;
            for (j = k; j < n; j++) { Ar[i * n + j] -= fr * Ar[k * n + j] - fi * Ai[k * n + j]; Ai[i * n + j] -= fr * Ai[k * n + j] + fi * Ar[k * n + j]; }
            for (j = 0; j < nr; j++) { Br[i * nr + j] -= fr * Br[k * nr + j] - fi * Bi[k * nr + j]; Bi[i * nr + j] -= fr * Bi[k * nr + j] + fi * Br[k * nr + j]; }
          }
        }
        for (k = n - 1; k >= 0; k--) {
          var dr2 = Ar[k * n + k], di2 = Ai[k * n + k], dd2 = dr2 * dr2 + di2 * di2, ir2 = dr2 / dd2, ii2 = -di2 / dd2;
          for (r = 0; r < nr; r++) {
            var sr = Br[k * nr + r], si = Bi[k * nr + r];
            for (j = k + 1; j < n; j++) { sr -= Ar[k * n + j] * Br[j * nr + r] - Ai[k * n + j] * Bi[j * nr + r]; si -= Ar[k * n + j] * Bi[j * nr + r] + Ai[k * n + j] * Br[j * nr + r]; }
            Br[k * nr + r] = sr * ir2 - si * ii2; Bi[k * nr + r] = sr * ii2 + si * ir2;
          }
        }
        return true;
      }
      // Network: nodes 0..N-1, elems [{ i, j, T }] (1-D two-ports, p-U), chambers [{ ch, patches:[{node, x0,x1,y,z,s, Zp:{r,i}}] }]
      // -> 2x2 T between node 0 (in) and node N-1 (out, U leaving).
      function tOfNet(N, elems, chambers, omega, rho, c0) {
        var Yr = new Float64Array(N * N), Yi = new Float64Array(N * N), e, q, i, j;
        function addY(a, b, z) { Yr[a * N + b] += z.r; Yi[a * N + b] += z.i; }
        for (q = 0; q < elems.length; q++) {
          e = elems[q]; var T = e.T, A = T[0][0], B = T[0][1], C = T[1][0], D = T[1][1];
          var iB = cinv(B);
          addY(e.i, e.i, cmul(D, iB));
          addY(e.i, e.j, csub(C, cmul(cmul(A, D), iB)));
          addY(e.j, e.i, cneg(iB));
          addY(e.j, e.j, cmul(A, iB));
        }
        for (q = 0; q < chambers.length; q++) {
          var chm = chambers[q], P = chm.patches, K = P.length;
          var Z = chamberZ3d(chm.ch, P, omega, rho, c0);
          var Ar = new Float64Array(K * K), Ai = new Float64Array(K * K), Br = new Float64Array(K * K), Bi = new Float64Array(K * K);
          for (i = 0; i < K * K; i++) { Ar[i] = Z.re[i]; Ai[i] = Z.im[i]; }
          for (i = 0; i < K; i++) { Ar[i * K + i] += P[i].Zp.r; Ai[i * K + i] += P[i].Zp.i; Br[i * K + i] = 1; }
          if (!csolve(K, Ar, Ai, K, Br, Bi)) continue;
          for (i = 0; i < K; i++) for (j = 0; j < K; j++) { Yr[P[i].node * N + P[j].node] += Br[i * K + j]; Yi[P[i].node * N + P[j].node] += Bi[i * K + j]; }
        }
        // Z-parameters at the two ports: Y p = I with I = e_in, e_out
        var Rr = new Float64Array(N * 2), Ri = new Float64Array(N * 2);
        Rr[0 * 2 + 0] = 1; Rr[(N - 1) * 2 + 1] = 1;
        if (!csolve(N, Yr, Yi, 2, Rr, Ri)) return null;
        var z11 = cx(Rr[0], Ri[0]), z12 = cx(Rr[1], Ri[1]), z21 = cx(Rr[(N - 1) * 2], Ri[(N - 1) * 2]), z22 = cx(Rr[(N - 1) * 2 + 1], Ri[(N - 1) * 2 + 1]);
        var i21 = cinv(z21);
        return [[cmul(z11, i21), cmul(csub(cmul(z11, z22), cmul(z12, z21)), i21)], [i21, cmul(z22, i21)]];
      }
      // perforate discretisation: midpoint rule, error ~ dx^2; 1.5 cm segments keep the TL within ~0.3 dB of the converged value
      function nsegOf(L) { return Math.max(6, Math.ceil(L / 0.015)); }
      // Builder helpers for the generic network
      function netBuilder() {
        var o = { N: 1, elems: [], chambers: [] };
        o.tube = function (a, L, area, omega, rho, c0, M) { if (L < 1e-6) return a; var b = o.N++; o.elems.push({ i: a, j: b, T: tTube(L, area, omega, rho, c0, M) }); return b; };
        // perforated tube run through chamber chm (local x from xa to xb), nseg segments, coupling via zPerforate
        o.perf = function (a, chm, xa, xb, y, z, d, nseg, sig, tw, dh, louv, omega, rho, c0, M) {
          var L = Math.abs(xb - xa), dx = L / nseg, area = PI * d * d / 4, s = d * Math.sqrt(PI) / 2, node = a, k;
          var zpc = zPerforate(omega, rho, c0, sig, tw, dh, M, louv), zs = rho * c0 / (PI * d * dx);
          for (k = 0; k < nseg; k++) {
            node = o.tube(node, (k === 0 ? 0.5 : 1) * dx, area, omega, rho, c0, M);
            var x0 = xa < xb ? xa + k * dx : xa - (k + 1) * dx, x1 = x0 + dx;
            if (x0 < 0) x0 = 0; if (x1 > chm.ch.L) x1 = chm.ch.L;
            chm.patches.push({ node: node, x0: x0, x1: x1, y: y, z: z, s: s, Zp: cx(zpc.r * zs, zpc.i * zs) });
          }
          return o.tube(node, 0.5 * dx, area, omega, rho, c0, M);
        };
        o.chamber = function (L, Ly, Lz) { var c = { ch: { L: L, Ly: Ly, Lz: Lz }, patches: [] }; o.chambers.push(c); return c; };
        return o;
      }
      // triflow3d: same gas path and parameters as tOfTriflow (US2182945 three-tube reverse flow), chambers as 3-D boxes.
      // Tube layout in the shell section (US3841435: symmetrically arranged tubes): inlet, return, outlet side by side on
      // the major axis at 1/6, 1/2, 5/6 of the width, on the minor-axis centre line.
      function tOfTriflow3d(p, omega, rho, c0) {
        var M = p.M || 0, sig = p.sigma, tw = p.tWall, dh = p.dHole, Lx = p.Lxover || 0.05;
        var aIn = PI * p.dIn * p.dIn / 4, aRet = PI * p.dRet * p.dRet / 4, aOut = PI * p.dOut * p.dOut / 4;
        var aShell = PI * p.aOval * p.bOval / 4;
        // Gas section of each chamber = shell section minus the tubes passing through it (same volumes as tOfTriflow),
        // as an equal-area rectangle with the oval's aspect ratio b/a. Tubes symmetrically arranged (US3841435):
        // inlet / return / outlet at y = Ly/6, Ly/2, 5Ly/6 on the major axis, z = Lz/2.
        function box(aNet) { var k = Math.sqrt(Math.max(aNet, 1e-6) / (p.aOval * p.bOval)); return [p.bOval * k, p.aOval * k]; }
        var b15 = box(aShell - aIn - aOut), b14 = box(aShell - aIn - aRet - aOut), b13 = box(aShell - aIn - aOut);
        var nb = netBuilder(), c15 = nb.chamber(p.L1, b15[0], b15[1]), c14 = nb.chamber(p.L2, b14[0], b14[1]), c13 = nb.chamber(p.L3, b13[0], b13[1]);
        var Ly = b14[0], zc = b14[1] / 2, yIn = Ly / 6, yRet = Ly / 2, yOut = 5 * b15[0] / 6, z15 = b15[1] / 2, z13 = b13[1] / 2;
        var n = 0;
        n = nb.tube(n, p.L1, aIn, omega, rho, c0, M);                                               // inlet, solid through ch15
        n = nb.perf(n, c14, 0, p.L2, yIn, zc, p.dIn, nsegOf(p.L2), sig, tw, dh, true, omega, rho, c0, M);     // louvred inlet in ch14
        n = nb.tube(n, Lx, aRet, omega, rho, c0, M);                                                // rear crossover (in ch13)
        var dNeck = p.dNeck || 0.0254, Lneck = (tw || 0.0015) + 0.85 * dNeck, Aneck = PI * dNeck * dNeck / 4;
        var r13 = p.rNeck || (0.5 * rho * c0 / Aneck * 0.05);
        c13.patches.push({ node: n, x0: 0, x1: 0, y: b13[0] / 3, z: z13, s: dNeck * Math.sqrt(PI) / 2, Zp: cx(r13, omega * rho * Lneck / Aneck) });
        n = nb.perf(n, c14, p.L2, 0, yRet, zc, p.dRet, nsegOf(p.L2), sig, tw, dh, true, omega, rho, c0, M);   // louvred return in ch14 (same cavity)
        n = nb.tube(n, Lx, aOut, omega, rho, c0, M);                                                // front crossover (in ch15)
        n = nb.perf(n, c15, 0, p.L1, yOut, z15, p.dOut, nsegOf(p.L1), sig, tw, dh, true, omega, rho, c0, M);   // louvred outlet in ch15
        n = nb.tube(n, p.L2 * 0.25, aOut, omega, rho, c0, M);
        n = nb.tube(n, p.L3, aOut, omega, rho, c0, M);
        var T = tOfNet(nb.N, nb.elems, nb.chambers, omega, rho, c0);
        return T || tOfTriflow(p, omega, rho, c0);
      }
      // perf_through3d: straight-through perforated tube on the shell axis, pinched partitions -> chambers (US3648803).
      function tOfPerfThrough3d(p, omega, rho, c0) {
        // gas section = shell minus tube (annulus), as an equal-area square; tube on the axis
        var Ls = Math.sqrt(Math.max(p.dShell * p.dShell - p.dTube * p.dTube, 1e-8) * PI / 4), fr = p.chFrac || [1], M = p.M || 0, i, n = 0;
        var nb = netBuilder();
        for (i = 0; i < fr.length; i++) {
          var Li = p.L * fr[i]; if (Li < 1e-4) continue;
          var c = nb.chamber(Li, Ls, Ls);
          n = nb.perf(n, c, 0, Li, Ls / 2, Ls / 2, p.dTube, nsegOf(Li), p.sigma, p.tWall, p.dHole, false, omega, rho, c0, M);
        }
        var T = tOfNet(nb.N, nb.elems, nb.chambers, omega, rho, c0);
        return T || tOfPerfThrough(p, omega, rho, c0);
      }

      // ---- IIR fit (Sanathanan–Koerner) + delay --------------------------------
      function skFit(freqs, Href, fs, na, nb) {
        // freqs in Hz, Href[k] = {r,i}, fit B/A, A monic of order na, B of order nb
        var n = freqs.length, i, k, it, pi2 = 2 * PI / fs;
        var a = [], b = [];
        for (i = 0; i <= na; i++) a[i] = i === 0 ? 1 : 0;
        for (i = 0; i <= nb; i++) b[i] = i === 0 ? 1 : 0;
        function evalPoly(c, z) { // c[0] + c[1] z^{-1} + ...
          var s = cx(0, 0), j, zj = cx(1, 0), zinv = cdiv(cx(1, 0), z);
          for (j = 0; j < c.length; j++) { s = cadd(s, cscale(zj, c[j])); zj = cmul(zj, zinv); }
          return s;
        }
        var col = na + nb + 1; // unknown: a1..ana, b0..bnb
        function solveLS(W) {
          // normal equations Re{ Φ^H Φ } x = Re{ Φ^H y }
          var m = 2 * n, ATA = [], ATb = [], r, c, t;
          for (r = 0; r < col; r++) { ATA[r] = []; for (c = 0; c < col; c++) ATA[r][c] = 0; ATb[r] = 0; }
          for (k = 0; k < n; k++) {
            var w = freqs[k] * pi2, z = cexpj(w), zinv, zj, wr = W ? W[k].r : 1, wi = W ? W[k].i : 0;
            // weight is complex 1/A or scalar
            var Hk = Href[k];
            // columns: for a1..ana: -H * z^{-i}, for b0..bnb: z^{-i}
            var phi = [];
            zinv = cdiv(cx(1, 0), z); zj = zinv;
            for (i = 1; i <= na; i++) { phi.push(cneg(cmul(Hk, zj))); zj = cmul(zj, zinv); }
            zj = cx(1, 0);
            for (i = 0; i <= nb; i++) { phi.push(zj); zj = cmul(zj, zinv); }
            // apply weight Wk (complex multiply)
            if (W) {
              for (i = 0; i < col; i++) phi[i] = cmul(phi[i], W[k]);
              Hk = cmul(Hk, W[k]);
            }
            // stack real/imag as two rows
            for (r = 0; r < col; r++) {
              ATb[r] += phi[r].r * Hk.r + phi[r].i * Hk.i;
              for (c = 0; c < col; c++) ATA[r][c] += phi[r].r * phi[c].r + phi[r].i * phi[c].i;
            }
          }
          // Gaussian elimination with partial pivoting
          var x = gaussSolve(ATA, ATb);
          if (!x) return null;
          var aa = [1], bb = [];
          for (i = 0; i < na; i++) aa.push(x[i]);
          for (i = 0; i <= nb; i++) bb.push(x[na + i]);
          return { a: aa, b: bb };
        }
        var sol = solveLS(null), last = sol;
        for (it = 0; it < 8; it++) {
          if (!sol) break;
          var W = [];
          for (k = 0; k < n; k++) {
            var z = cexpj(freqs[k] * pi2), A = evalPoly(sol.a, z);
            W[k] = cinv(A);
          }
          last = sol;
          sol = solveLS(W);
        }
        if (!sol) sol = last;
        return sol;
      }
      function gaussSolve(A, b) {
        var n = b.length, i, j, k, p, t, x = [];
        var M = [];
        for (i = 0; i < n; i++) { M[i] = A[i].slice(); M[i][n] = b[i]; }
        for (i = 0; i < n; i++) {
          p = i;
          for (k = i + 1; k < n; k++) if (Math.abs(M[k][i]) > Math.abs(M[p][i])) p = k;
          if (Math.abs(M[p][i]) < 1e-14) return null;
          if (p !== i) { t = M[i]; M[i] = M[p]; M[p] = t; }
          for (k = i + 1; k < n; k++) {
            var f = M[k][i] / M[i][i];
            for (j = i; j <= n; j++) M[k][j] -= f * M[i][j];
          }
        }
        for (i = n - 1; i >= 0; i--) {
          t = M[i][n];
          for (j = i + 1; j < n; j++) t -= M[i][j] * x[j];
          x[i] = t / M[i][i];
          if (!isFinite(x[i])) return null;
        }
        return x;
      }

      function iirImpulseStable(a, b, n) {
        var nA = a.length - 1, nB = b.length - 1, x = [], i, k, acc, e = 0, last = 0;
        for (i = 0; i < n; i++) {
          acc = (i <= nB ? b[i] : 0);
          for (k = 1; k <= nA && k <= i; k++) acc -= a[k] * x[i - k];
          x[i] = acc;
          e += acc * acc;
          if (!isFinite(acc) || e > 1e8) return false;
        }
        // late energy should decay
        var e1 = 0, e2 = 0, h = (n / 2) | 0;
        for (i = 0; i < h; i++) e1 += x[i] * x[i];
        for (i = h; i < n; i++) e2 += x[i] * x[i];
        return e2 < e1 * 4 + 1e-8;
      }
      function shrinkPoles(a, rho) {
        // A(z) -> A(ρ z): a[k] *= ρ^k  pulls poles inward
        var out = [a[0]], i, p = rho;
        for (i = 1; i < a.length; i++) { out[i] = a[i] * p; p *= rho; }
        return out;
      }


      function pronyFit(h, na, nb) {
        var N = h.length, n0 = Math.max(na, nb), i, n, r, c, acc;
        if (N < n0 + na + 4) return null;
        var col = na, ATA = [], ATb = [];
        for (r = 0; r < col; r++) { ATA[r] = []; for (c = 0; c < col; c++) ATA[r][c] = 0; ATb[r] = 0; }
        for (n = n0; n < N; n++) {
          for (r = 0; r < na; r++) {
            var hr = h[n - r - 1] || 0;
            ATb[r] -= hr * h[n];
            for (c = 0; c < na; c++) ATA[r][c] += hr * (h[n - c - 1] || 0);
          }
        }
        var x = gaussSolve(ATA, ATb);
        if (!x) return null;
        var a = [1], b = [];
        for (i = 0; i < na; i++) a.push(x[i]);
        for (i = 0; i <= nb; i++) {
          acc = h[i] || 0;
          for (r = 1; r <= na && r <= i; r++) acc += a[r] * (h[i - r] || 0);
          b[i] = acc;
        }
        return { a: a, b: b };
      }

      // Time-domain Steiglitz–McBride: IIR fit of a causal impulse. Used for the
      // tail after a short FIR head (a plain SK/Prony on the full IR is a poor
      // fit when residual onset is delayed).
      function filter1A(x, a) {
        var y = [], n, k, acc, nA = a.length - 1;
        for (n = 0; n < x.length; n++) {
          acc = x[n];
          for (k = 1; k <= nA && k <= n; k++) acc -= a[k] * y[n - k];
          y[n] = acc;
        }
        return y;
      }
      function steiglitzMcBride(h, na, nb, iters) {
        var sol = pronyFit(h, na, nb), it, n, r, c, N, x, u, d, col, ATA, ATb, xv, a, b, phi, yv;
        if (!sol) return null;
        N = h.length;
        x = [];
        for (n = 0; n < N; n++) x[n] = n === 0 ? 1 : 0;
        iters = iters || 8;
        col = na + nb + 1;
        for (it = 0; it < iters; it++) {
          u = filter1A(x, sol.a);
          d = filter1A(h, sol.a);
          ATA = []; ATb = [];
          for (r = 0; r < col; r++) { ATA[r] = []; for (c = 0; c < col; c++) ATA[r][c] = 0; ATb[r] = 0; }
          for (n = 0; n < N; n++) {
            phi = [];
            for (c = 1; c <= na; c++) phi.push(-(n - c >= 0 ? d[n - c] : 0));
            for (c = 0; c <= nb; c++) phi.push(n - c >= 0 ? u[n - c] : 0);
            yv = d[n];
            for (r = 0; r < col; r++) {
              ATb[r] += phi[r] * yv;
              for (c = 0; c < col; c++) ATA[r][c] += phi[r] * phi[c];
            }
          }
          for (r = 0; r < col; r++) ATA[r][r] += 1e-10;
          xv = gaussSolve(ATA, ATb);
          if (!xv) break;
          a = [1]; b = [];
          for (n = 0; n < na; n++) a.push(xv[n]);
          for (n = 0; n <= nb; n++) b.push(xv[na + n]);
          sol = { a: a, b: b };
        }
        return sol;
      }

      function unwrapPhase(ph) {
        var i, out = [ph[0]], tw = 0;
        for (i = 1; i < ph.length; i++) {
          var d = ph[i] - ph[i - 1];
          while (d > Math.PI) { tw -= 2 * Math.PI; d -= 2 * Math.PI; }
          while (d < -Math.PI) { tw += 2 * Math.PI; d += 2 * Math.PI; }
          out[i] = ph[i] + tw;
        }
        return out;
      }
      function groupDelaySamples(freqs, H, fs) {
        // linear-phase slope of arg(H) vs f, in samples
        var ph = [], i;
        for (i = 0; i < H.length; i++) ph.push(Math.atan2(H[i].i, H[i].r));
        ph = unwrapPhase(ph);
        // fit slope on mid band (skip DC and near-Nyquist)
        var i0 = Math.max(2, (H.length * 0.05) | 0), i1 = Math.min(H.length - 2, (H.length * 0.4) | 0);
        var sx = 0, sy = 0, sxx = 0, sxy = 0, n = 0, x, y;
        for (i = i0; i < i1; i++) {
          if (cabs(H[i]) < 1e-4) continue;
          x = 2 * PI * freqs[i] / fs;
          y = ph[i];
          sx += x; sy += y; sxx += x * x; sxy += x * y; n++;
        }
        if (n < 4) return 0;
        var den = n * sxx - sx * sx;
        if (Math.abs(den) < 1e-12) return 0;
        var slope = (n * sxy - sx * sy) / den; // d phase / d(ω/fs) = -delay_samples
        var d = -slope;
        if (d < 0) d = 0;
        if (d > 240) d = 240;
        return d;
      }

      function fitTransfer(freqs, H, fs, delayHint, preferIir, opt) {
        opt = opt || {};
        var rev2 = !!(preferIir && opt.rev >= 2);
        var delay = (delayHint !== undefined && delayHint !== null) ? Math.round(delayHint) : Math.round(groupDelaySamples(freqs, H, fs));
        if (delay < 0) delay = 0;
        if (delay > 240) delay = 240;
        var i, Hd = [];
        for (i = 0; i < H.length; i++) {
          var w = 2 * PI * freqs[i] / fs;
          Hd[i] = cmul(H[i], cexpj(w * delay));
          // Decimated-rate fits must be quiet at the design Nyquist or the
          // IIR/FIR leaks just above 0.42*fs (e.g. 6 kHz at fsFit=12 kHz).
          if (rev2 && fs) {
            var frac = freqs[i] / fs, tw = 1;
            if (frac >= 0.45) tw = 0;
            else if (frac > 0.38) tw = 0.5 + 0.5 * Math.cos(PI * (frac - 0.38) / 0.07);
            if (tw !== 1) Hd[i] = cscale(Hd[i], tw);
          }
        }
        var na = 8, nb = 8, sol = skFit(freqs, Hd, fs, na, nb);
        if (!sol) sol = { a: [1], b: [0] };
        function qual(filt) {
          var e = 0, n = 0, mx = 0, mxTL = 0, underTL = 0, hfMax = 0, hfUnderTL = 0, i, f, w, A, B, mag, tmag, dlt, tTL, mTL;
          for (i = 0; i < freqs.length; i++) {
            f = freqs[i];
            var fHi = rev2 ? Math.min(8000, 0.42 * fs) : 2000;
            if (f < 20 || f > fHi) continue;
            var mid = f <= 2000;
            tmag = Math.sqrt(Hd[i].r * Hd[i].r + Hd[i].i * Hd[i].i);
            mag = 0;
            if (filt.typ === 'iir' || filt.typ === 'hyb') {
              w = 2 * Math.PI * f / fs;
              A = { r: 0, i: 0 }; B = { r: 0, i: 0 };
              var zk = { r: 1, i: 0 }, zinv = { r: Math.cos(-w), i: Math.sin(-w) }, k, zr, zi;
              if (filt.a && filt.b) {
                for (k = 0; k < filt.a.length; k++) {
                  A.r += filt.a[k] * zk.r; A.i += filt.a[k] * zk.i;
                  zr = zk.r * zinv.r - zk.i * zinv.i; zi = zk.r * zinv.i + zk.i * zinv.r; zk.r = zr; zk.i = zi;
                }
                zk = { r: 1, i: 0 };
                for (k = 0; k < filt.b.length; k++) {
                  B.r += filt.b[k] * zk.r; B.i += filt.b[k] * zk.i;
                  zr = zk.r * zinv.r - zk.i * zinv.i; zi = zk.r * zinv.i + zk.i * zinv.r; zk.r = zr; zk.i = zi;
                }
              }
              var den = A.r * A.r + A.i * A.i;
              var iirR = 0, iirI = 0;
              if (den >= 1e-18) { iirR = (B.r * A.r + B.i * A.i) / den; iirI = (B.i * A.r - B.r * A.i) / den; }
              if (filt.typ === 'hyb') {
                var hr = 0, hi = 0, td = filt.tailDelay || (filt.fir ? filt.fir.length : 0);
                for (k = 0; k < filt.fir.length; k++) {
                  hr += filt.fir[k] * Math.cos(-w * k);
                  hi += filt.fir[k] * Math.sin(-w * k);
                }
                var pr = Math.cos(-w * td), pi = Math.sin(-w * td);
                hr += iirR * pr - iirI * pi;
                hi += iirR * pi + iirI * pr;
                mag = Math.sqrt(hr * hr + hi * hi);
              } else {
                mag = Math.sqrt(iirR * iirR + iirI * iirI);
              }
            }
            dlt = Math.abs(mag - tmag);
            tTL = -20 * Math.log(Math.max(1e-6, tmag)) / Math.LN10;
            mTL = -20 * Math.log(Math.max(1e-6, mag)) / Math.LN10;
            if (mid) {
              e += dlt; n++;
              if (dlt > mx) mx = dlt;
              if ((!rev2 || tTL < 18) && Math.abs(tTL - mTL) > mxTL) mxTL = Math.abs(tTL - mTL);
              if (tTL - mTL > underTL) underTL = tTL - mTL;
            } else if (rev2) {
              if (dlt > hfMax) hfMax = dlt;
              if (tTL < 25 && tTL - mTL > hfUnderTL) hfUnderTL = tTL - mTL;
            }
          }
          return { mae: n ? e / n : 1, max: mx, maxTL: mxTL, underTL: underTL, hfMax: hfMax, hfUnderTL: hfUnderTL };
        }
        function lowErr(filt) { return qual(filt).mae; }
        function cheapOk(q) {
          if (!q) return false;
          if (rev2) {
            // Mid-band (20-2 kHz) stays tight. Above 2 kHz, ignore deep-stopband TL-ratio
            // errors (32 vs 47 dB is inaudible) but reject |S| leaks and under-attenuation
            // where the FD can is not already a deep notch (TL < 25 dB).
            // Mid-band tightness plus an HF under-TL cap where the FD can is not
            // already a deep notch. Do not fail a mid-band-good fit on stopband |S|
            // shape (a 7 kHz notch shift is not a loudness leak).
            return q.mae <= 0.10 && q.max <= 0.18 && q.maxTL <= 6 && (q.underTL == null || q.underTL <= 6)
              && (q.hfUnderTL == null || q.hfUnderTL <= 3.5);
          }
          return q.mae <= 0.10 && q.max <= 0.18 && q.maxTL <= 6;
        }
        function stabilize(sol) {
          var guard = 0;
          while (sol && !iirImpulseStable(sol.a, sol.b, 1024) && guard < 10) {
            sol.a = shrinkPoles(sol.a, 0.96);
            guard++;
          }
          return sol && iirImpulseStable(sol.a, sol.b, 1024) ? sol : null;
        }
        var iirOK = stabilize(sol);
        var iirFilt = iirOK ? { typ: 'iir', delay: delay, a: iirOK.a, b: iirOK.b, fir: null } : null;
        var err = iirFilt ? lowErr(iirFilt) : 99;
        if (preferIir && err > 0.05) {
          var orders = [8, 12, 16], oi, cand, e2, best = iirFilt, bestE = err;
          for (oi = 0; oi < orders.length; oi++) {
            cand = stabilize(skFit(freqs, Hd, fs, orders[oi], orders[oi]));
            if (!cand) continue;
            e2 = lowErr({ typ: 'iir', a: cand.a, b: cand.b });
            if (e2 < bestE) { bestE = e2; best = { typ: 'iir', delay: delay, a: cand.a, b: cand.b, fir: null }; }
          }
          // Prony on the delay-compensated impulse, then one SK refine
          var hIR = firFromH(freqs, Hd, fs, 256, rev2);
          for (oi = 0; oi < orders.length; oi++) {
            cand = pronyFit(hIR, orders[oi], orders[oi]);
            cand = stabilize(cand);
            if (!cand) continue;
            e2 = lowErr({ typ: 'iir', a: cand.a, b: cand.b });
            if (e2 < bestE) { bestE = e2; best = { typ: 'iir', delay: delay + (hIR._rot || 0), a: cand.a, b: cand.b, fir: null }; }
          }
          iirFilt = best; err = bestE;
          // Hybrid: short FIR head (main arrival) + low-order IIR on the delayed tail.
          // A few-pole IIR on the whole delay-compensated IR is a poor fit when the
          // residual still starts tens of samples late; the head+tail split is cheap
          // and keeps LF |S| (and TL) close to the frequency-domain can.
          var href = firFromH(freqs, Hd, fs, 512, rev2);
          var heads, hybOrd;
          if (rev2) {
            var isRefl = opt.name === 'S11' || opt.name === 'S22';
            heads = isRefl ? [16, 24, 32, 48, 64, 96, 128, 160] : [32, 48, 64, 80, 96, 128, 160, 192, 256];
            hybOrd = isRefl ? [4, 6, 8] : [4, 6, 8];
          } else {
            heads = [32, 48, 64, 96, 128, 192, 256, 320];
            hybOrd = [4, 6, 8, 12];
          }
          var hLen, tail, hyb, qH, found = false, bestQ = { mae: 99, max: 99, maxTL: 99, underTL: 99 };
          for (oi = 0; oi < heads.length && !found; oi++) {
            hLen = heads[oi];
            if (!href || href.length <= hLen + 16) continue;
            tail = href.slice(hLen);
            var oj;
            for (oj = 0; oj < hybOrd.length; oj++) {
              cand = stabilize(steiglitzMcBride(tail, hybOrd[oj], hybOrd[oj], 8));
              if (!cand) continue;
              hyb = { typ: 'hyb', delay: delay + (href._rot || 0), fir: href.slice(0, hLen), a: cand.a, b: cand.b, tailDelay: hLen };
              qH = qual(hyb);
              if (cheapOk(qH)) { best = hyb; bestE = qH.mae; bestQ = qH; found = true; break; }
              if (rev2) {
                function score(q) { return (q.underTL || 0) + (q.maxTL || 0) + 8 * (q.mae || 0); }
                if (score(qH) < score(bestQ) - 1e-6) { bestQ = qH; bestE = qH.mae; best = hyb; }
              } else if (qH.maxTL < bestQ.maxTL || (qH.maxTL === bestQ.maxTL && qH.mae < bestQ.mae)) {
                bestQ = qH; bestE = qH.mae; best = hyb;
              }
            }
          }
          iirFilt = best; err = bestE;
        }
        var useFir = !iirFilt || (preferIir ? !cheapOk(qual(iirFilt)) : err > 0.08);
        // rev>=2 may still fall back to a FIR, but a shorter one designed at fsFit (often fs/2).
        if (useFir) {
          var fir = firFromH(freqs, Hd, fs, rev2 ? 256 : 512, rev2);
          var firFilt = { typ: 'fir', delay: delay + (fir._rot || 0), fir: fir, a: null, b: null };
          if (firFilt.delay > 240) {
            var drop = firFilt.delay - 240;
            if (drop < fir.length) fir = fir.slice(drop);
            firFilt.delay = 240;
            firFilt.fir = fir;
          }
          return firFilt;
        }
        return iirFilt;
      }

      function firFromH(freqs, H, fs, nt, useMaxEnergy) {
        // Frequency-sampling FIR via IFFT of a Hermitian spectrum. Rotates the impulse
        // so the maximum-energy nt-tap window starts at 0; the caller adds that rotation to delay.
        var N = 2048, i, k, re, h = [], nF = freqs.length;
        var Hr = new Array(N), Hi = new Array(N);
        for (i = 0; i < N; i++) { Hr[i] = 0; Hi[i] = 0; }
        for (i = 0; i <= N / 2; i++) {
          var f = i * fs / N, Hv;
          if (f <= freqs[0]) Hv = H[0];
          else if (f >= freqs[nF - 1]) Hv = H[nF - 1];
          else {
            k = 0; while (k < nF - 1 && freqs[k + 1] < f) k++;
            var tt = (f - freqs[k]) / Math.max(1e-9, freqs[k + 1] - freqs[k]);
            Hv = cadd(cscale(H[k], 1 - tt), cscale(H[k + 1], tt));
          }
          if (useMaxEnergy) {
            var frac = f / fs, tw = 1;
            if (frac >= 0.45) tw = 0;
            else if (frac > 0.38) tw = 0.5 + 0.5 * Math.cos(PI * (frac - 0.38) / 0.07);
            if (tw !== 1) Hv = cscale(Hv, tw);
          }
          Hr[i] = Hv.r; Hi[i] = Hv.i;
          if (i > 0 && i < N / 2) { Hr[N - i] = Hv.r; Hi[N - i] = -Hv.i; }
        }
        // IFFT (real part only; Hermitian input)
        for (i = 0; i < N; i++) {
          re = 0;
          for (k = 0; k < N; k++) {
            var ang = 2 * PI * i * k / N;
            re += Hr[k] * Math.cos(ang) + Hi[k] * Math.sin(ang);
          }
          h[i] = re / N;
        }
        // Start at the first arrival (not the loudest later echo) so peeled delay stays honest.
        // rev>=2: if that window holds little energy (common for S11, where the IFFT peak
        // sits just before t=0 and h[0] may miss the 0.08*peak wrap-search trigger), fall
        // back to the maximum-energy nt-tap window. Wrap-around windows keep delay 0, as
        // the first-arrival path already does when rot===0 finds a pre-t=0 peak.
        var peak = 0, rot = 0, thr;
        for (i = 0; i < N; i++) if (Math.abs(h[i]) > peak) peak = Math.abs(h[i]);
        thr = 0.08 * peak;
        for (i = 0; i < (N / 2); i++) if (Math.abs(h[i]) >= thr) { rot = i; break; }
        if (rot === 0) {
          for (i = N - ((N / 8) | 0); i < N; i++) if (Math.abs(h[i]) >= thr) { rot = i; break; }
        }
        if (useMaxEnergy) {
          var eArr = 0, eMax = 0, rotE = 0, j, acc, drop, addS;
          for (j = 0; j < nt; j++) eArr += h[(j + rot) % N] * h[(j + rot) % N];
          acc = eArr; eMax = eArr; rotE = rot;
          for (i = 1; i < N; i++) {
            drop = h[(rot + i - 1) % N]; addS = h[(rot + i - 1 + nt) % N];
            acc += addS * addS - drop * drop;
            if (acc > eMax) { eMax = acc; rotE = (rot + i) % N; }
          }
          if (eArr < 0.5 * eMax) {
            // First-arrival window missed the main energy (typical for S11: IFFT
            // peak just before t=0). Align to that peak with a short pre-roll so
            // short hybrid heads still see the arrival. A long max-energy window
            // would bury the peak at the end, under the taper.
            var pki = 0, pkA = 0, pre;
            for (i = 0; i < N; i++) if (Math.abs(h[i]) > pkA) { pkA = Math.abs(h[i]); pki = i; }
            pre = 8;
            rot = (pki - pre + N) % N;
          }
        }
        var out = [], w, add = (rot > N / 2) ? 0 : rot;
        for (i = 0; i < nt; i++) {
          w = (i > nt * 0.85) ? 0.5 + 0.5 * Math.cos(PI * (i - nt * 0.85) / (nt * 0.15)) : 1;
          out[i] = h[(i + rot) % N] * w;
        }
        out._rot = add;
        return out;
      }

      // Passivity: if max power gain > 1, scale all Sij by 1/sqrt(gmax)
      function scalePassive(Ss, area1, area2) {
        var i, gmax = 1;
        for (i = 0; i < Ss.length; i++) {
          var S = Ss[i], t21 = (area2 / area1) * (S.S21.r * S.S21.r + S.S21.i * S.S21.i);
          var t12 = (area1 / area2) * (S.S12.r * S.S12.r + S.S12.i * S.S12.i);
          var r11 = S.S11.r * S.S11.r + S.S11.i * S.S11.i;
          var r22 = S.S22.r * S.S22.r + S.S22.i * S.S22.i;
          // loose bound: reflection + transmission power
          var g = r11 + t21;
          if (g > gmax) gmax = g;
          g = r22 + t12;
          if (g > gmax) gmax = g;
        }
        if (gmax <= 1.02) return 1;
        var sc = 1 / Math.sqrt(gmax * 1.01);
        for (i = 0; i < Ss.length; i++) {
          var S2 = Ss[i];
          S2.S11 = cscale(S2.S11, sc); S2.S12 = cscale(S2.S12, sc);
          S2.S21 = cscale(S2.S21, sc); S2.S22 = cscale(S2.S22, sc);
        }
        return sc;
      }

      function gas(T, M) {
        T = T || 650;
        var rho = 101325 / (287 * T), c0 = Math.sqrt(1.30 * 287 * T);
        return { rho: rho, c0: c0, T: T, M: M || 0 };
      }

      function sampleCan(kind, p, fs, nF) {
        nF = nF || 256;
        var g = gas(p.T, p.M), i, freqs = [], Ss = [], TLs = [];
        var tri = kind === 'triflow' || kind === 'triflow3d';
        var a1 = tri ? PI * p.dIn * p.dIn / 4 : PI * p.dTube * p.dTube / 4;
        var a2 = tri ? PI * p.dOut * p.dOut / 4 : a1;
        var fMin = 10, fMax = 0.45 * fs;
        for (i = 0; i < nF; i++) {
          // log grid so low-frequency TL (chamber / Helmholtz) is well sampled
          var f = fMin * Math.pow(fMax / fMin, (i + 0.5) / nF);
          freqs.push(f);
          var T = tOfCan(kind, p, 2 * PI * f, g.rho, g.c0);
          var S = tToS(T, a1, a2, g.rho, g.c0, p.rev);
          Ss.push(S);
          TLs.push(tlOfS(S, a1, a2));
        }
        var sc = scalePassive(Ss, a1, a2);
        if (p.rev >= 1) {
          for (i = 0; i < Ss.length; i++) TLs[i] = tlOfS(Ss[i], a1, a2);
        }
        return { freqs: freqs, S: Ss, TL: TLs, area1: a1, area2: a2, gas: g, scale: sc };
      }

      // rev 4 (v14.1, 3-D cans): the 3-D responses have narrow cross-mode peaks that a 256-point log grid under-samples
      // (75 Hz apart at 3 kHz), so the rev-3 fitter loses up to ~10 dB of band energy. Their impulse responses are short
      // (99 % of the energy within ~150 samples at fs/4), so rev 4 takes the exact causal IR from a dense linear grid
      // (inverse DFT, same passivity scaling and same 0.38-0.45 fsFit taper as rev 3) and splits it into the same hybrid
      // filter: FIR head + order-8 IIR tail (Steiglitz-McBride). Run-time size is fixed and no larger than the rev-3 cans:
      // 16-tap head + order 8 on all four paths (34 multiply-adds per fsFit sample; rev 3 uses 30-42). Decimation 4 and
      // interpolation as rev 3. Fit error, octave band power vs the frequency-domain model: mostly within 1.5 dB, worst 2.5 dB.
      var CAN4_NFFT = 1024;
      function compileCan4(kind, p, fs) {
        var decim = 4, fsFit = fs / decim, N = CAN4_NFFT, g = gas(p.T, p.M), k, n, i;
        var tri = kind === 'triflow' || kind === 'triflow3d';
        var a1 = tri ? PI * p.dIn * p.dIn / 4 : PI * p.dTube * p.dTube / 4, a2 = tri ? PI * p.dOut * p.dOut / 4 : a1;
        var freqs = [], Ss = [], TLs = [];
        for (k = 0; k <= N / 2; k++) {
          var f = k === 0 ? 0.25 * fsFit / N : k * fsFit / N;
          freqs.push(f);
          Ss.push(tToS(tOfCan(kind, p, 2 * PI * f, g.rho, g.c0), a1, a2, g.rho, g.c0, p.rev));
        }
        var sc = scalePassive(Ss, a1, a2);
        for (k = 0; k < Ss.length; k++) TLs.push(tlOfS(Ss[k], a1, a2));
        var names = ['S11', 'S12', 'S21', 'S22'], filt = {};
        for (i = 0; i < 4; i++) {
          var nm = names[i], h = new Float64Array(N), re = [], im = [];
          for (k = 0; k <= N / 2; k++) {
            var fr = freqs[k] / fsFit, tw = fr >= 0.45 ? 0 : (fr > 0.38 ? 0.5 + 0.5 * Math.cos(PI * (fr - 0.38) / 0.07) : 1);
            if (k === N / 2) tw = 0;
            re.push(Ss[k][nm].r * tw); im.push(k === 0 ? 0 : Ss[k][nm].i * tw);
          }
          for (n = 0; n < N; n++) {
            var acc = re[0];
            for (k = 1; k < N / 2; k++) { var w = 2 * PI * k * n / N; acc += 2 * (re[k] * Math.cos(w) - im[k] * Math.sin(w)); }
            h[n] = acc / N;
          }
          // pure delay: up to the first sample carrying 0.1 % of the energy (no earlier than 0)
          var tot = 0, cum = 0, d = 0;
          for (n = 0; n < N; n++) tot += h[n] * h[n];
          for (n = 0; n < N; n++) { cum += h[n] * h[n]; if (cum > 1e-3 * tot) { d = n; break; } }
          if (d > 240) d = 240;
          var hLen = 16, ord = 8;
          var head = Array.prototype.slice.call(h, d, d + hLen), tail = Array.prototype.slice.call(h, d + hLen, Math.min(N, d + hLen + 768));
          var sol = steiglitzMcBride(tail, ord, ord, 8), guard = 0;
          while (sol && !iirImpulseStable(sol.a, sol.b, 1024) && guard < 10) { sol.a = shrinkPoles(sol.a, 0.96); guard++; }
          if (!sol || !iirImpulseStable(sol.a, sol.b, 1024)) sol = { a: [1], b: [0] };
          filt[nm] = { typ: 'hyb', delay: d, fir: head, a: sol.a, b: sol.b, tailDelay: hLen };
        }
        // decimated path output interpolation lags one fsFit sample (as rev 3)
        for (i = 0; i < 4; i++) if (filt[names[i]].delay >= 1) filt[names[i]].delay -= 1;
        var out = { kind: kind, p: p, fs: fs, fsFit: fsFit, decim: decim, filt: filt, TL: TLs, freqs: freqs, area1: a1, area2: a2, scale: sc, gas: g };
        stabilizeCan(out);
        return out;
      }

      function compileCan(kind, p, fs) {
        var decim = (p && p.rev >= 2) ? 4 : 1;
        var fsFit = fs / decim;
        if (p && p.rev >= 4) return compileCan4(kind, p, fs);
        var samp = sampleCan(kind, p, fsFit, 256);
        var names = ['S11', 'S12', 'S21', 'S22'], i, k, filt = {};
        var path = 0;
        if (kind === 'triflow' || kind === 'triflow3d') path = p.L1 + 2 * p.L2 + 2 * (p.Lxover || 0) + p.L1 + 0.25 * p.L2 + p.L3;
        else path = p.L || 0;
        var dTx = samp.gas.c0 > 0 ? path / samp.gas.c0 * fsFit : 0;
        for (i = 0; i < 4; i++) {
          var H = [];
          for (k = 0; k < samp.S.length; k++) H.push(samp.S[k][names[i]]);
          // S21/S12 carry the through-path delay; S11/S22 start with a near-end reflection (hint 0, let group delay add far-end).
          var hint = (names[i] === 'S21' || names[i] === 'S12') ? dTx : 0;
          filt[names[i]] = fitTransfer(samp.freqs, H, fsFit, hint, p.rev >= 1, { rev: p.rev || 0, name: names[i] });
        }
        // rev>=3: the decimated path's output interpolation lags one fsFit sample; take it
        // off the fitted pure delay so the can adds no latency over the full-rate fit.
        if (decim > 1 && p.rev >= 3) {
          for (i = 0; i < 4; i++) if (filt[names[i]].delay >= 1) filt[names[i]].delay -= 1;
        }
        var out = { kind: kind, p: p, fs: fs, fsFit: fsFit, decim: decim, filt: filt, TL: samp.TL, freqs: samp.freqs, area1: samp.area1, area2: samp.area2, scale: samp.scale, gas: samp.gas };
        if (p.rev >= 1) stabilizeCan(out);
        return out;
      }

      // Runtime filter state
      function makeState(comp) {
        var st = { decim: (comp && comp.decim) || 1, rev: (comp && comp.p && comp.p.rev) || 0, _ph: 0, _o1: 0, _o2: 0, _o1p: 0, _o2p: 0, _s1: 0, _s2: 0 }, n, f, names = ['S11', 'S12', 'S21', 'S22'];
        for (n = 0; n < 4; n++) {
          f = comp.filt[names[n]];
          st[names[n]] = {
            typ: f.typ, delay: f.delay, a: f.a, b: f.b, fir: f.fir,
            tailDelay: f.tailDelay || 0,
            xd: [], yd: [], fd: [], di: 0
          };
          var cap = (f.delay | 0) + (f.tailDelay | 0) + 4 + (f.fir ? f.fir.length + 2 : 0) + (f.a ? f.a.length + 2 : 4);
          if (cap < 16) cap = 16;
          st[names[n]].xd = new Float64Array(cap);
          st[names[n]].yd = new Float64Array(cap);
          if (f.typ === 'fir' || f.typ === 'hyb') {
            st[names[n]].fd = new Float64Array(f.fir.length);
            st[names[n]].fi = 0;
          }
        }
        return st;
      }
      function filtSample(st, x) {
        if (!isFinite(x)) x = 0;
        var d = st.delay | 0, cap = st.xd.length, y, i, acc;
        // write input into delay line
        st.di = (st.di + 1) % cap;
        st.xd[st.di] = x;
        var xi = st.xd[(st.di - d + cap) % cap];
        if (st.typ === 'fir' || st.typ === 'hyb') {
          var fl = st.fir.length, fi = st.fi | 0, fd = st.fd, fir = st.fir, j;
          fd[fi] = xi;
          acc = 0; j = fi;
          for (i = 0; i < fl; i++) { acc += fir[i] * fd[j]; if (--j < 0) j = fl - 1; }
          st.fi = fi + 1; if (st.fi >= fl) st.fi = 0;
          y = acc;
        } else y = 0;
        if (st.typ === 'iir' || st.typ === 'hyb') {
          var a = st.a, b = st.b, na = a.length - 1, nb = b.length - 1;
          var td = st.typ === 'hyb' ? (st.tailDelay | 0) : 0;
          var u = st.xd[(st.di - d - td + cap * 8) % cap];
          acc = b[0] * u;
          for (i = 1; i <= nb; i++) acc += b[i] * st.xd[(st.di - d - td - i + cap * 8) % cap];
          for (i = 1; i <= na; i++) acc -= a[i] * st.yd[(st.di - i + cap) % cap];
          if (st.typ === 'iir') y = acc;
          else y += acc;
          if (!isFinite(acc)) acc = 0;
          st.yd[st.di] = acc; // IIR state only
        } else {
          if (!isFinite(y)) y = 0;
          st.yd[st.di] = y;
        }
        if (!isFinite(y)) y = 0;
        return y;
      }
      function applyCanFull(st, p1in, p2in) {
        var y11 = filtSample(st.S11, p1in), y12 = filtSample(st.S12, p2in);
        var y21 = filtSample(st.S21, p1in), y22 = filtSample(st.S22, p2in);
        var o1 = y11 + y12, o2 = y21 + y22;
        if (!isFinite(o1)) o1 = 0;
        if (!isFinite(o2)) o2 = 0;
        return { o1: o1, o2: o2 };
      }
      // rev>=2: run the four Sij filters at fs/decim (designed at fsFit). Inputs
      // are boxcar-averaged, outputs linearly interpolated. State is per can
      // (banks are not shared).
      // v15: one can with two prebuilt gas states (idle, cruise). Steady: only the selected filter runs. On a change the
      // new filter runs alongside for DUAL_WARM samples (fills its state from the same waves), then a DUAL_XF-sample linear
      // crossfade; after that only the new one runs (the synth swaps the plain state back in at the next block, so the steady
      // path is exactly the single-filter one). CPU is doubled for ~32 ms per change only.
      var DUAL_WARM = 512, DUAL_XF = 1024;
      function applyDual(st, p1in, p2in) {
        var ya = applyCan(st.cur, p1in, p2in);
        if (!st.nxt) return ya;
        var yb = applyCan(st.nxt, p1in, p2in), k = st.k++;
        if (k < DUAL_WARM) return ya;
        var t = (k - DUAL_WARM + 1) / DUAL_XF;
        if (t >= 1) { st.cur = st.nxt; st.nxt = null; st.done = true; return yb; }
        return { o1: ya.o1 + t * (yb.o1 - ya.o1), o2: ya.o2 + t * (yb.o2 - ya.o2) };
      }
      function applyCan(st, p1in, p2in) {
        var d = st.decim | 0, ph, t, y;
        if (d <= 1) return applyCanFull(st, p1in, p2in);
        st._s1 = (st._s1 || 0) + p1in;
        st._s2 = (st._s2 || 0) + p2in;
        ph = (st._ph | 0) + 1;
        if (ph < d) {
          st._ph = ph;
          t = ph / d;
          return { o1: st._o1p + t * (st._o1 - st._o1p), o2: st._o2p + t * (st._o2 - st._o2p) };
        }
        y = applyCanFull(st, st._s1 / d, st._s2 / d);
        st._s1 = 0; st._s2 = 0; st._ph = 0;
        st._o1p = st._o1; st._o2p = st._o2;
        st._o1 = y.o1; st._o2 = y.o2;
        // rev>=3: continuous linear interpolation (t = 0 at the update). rev 2 returned the
        // new value here and then restarted from the previous one, a sawtooth at fs/decim
        // (a ~12 kHz whine at 48 kHz). The one decimated sample of latency is taken off the
        // fitted bulk delay in compileCan.
        if (st.rev >= 3) return { o1: st._o1p, o2: st._o2p };
        return y;
      }

      // After the IIR tail is fitted, a long two-port impulse must still decay.
      // Poles can sit near the unit circle (the short iirImpulseStable check misses
      // slow growth) and then the waveguide loop at 44.1 / 96 kHz runs away.
      function canGrows(comp) {
        var n = 16384, port, i, o, e1, e2, en, st;
        for (port = 0; port < 2; port++) {
          st = makeState(comp);
          e1 = 0; e2 = 0;
          for (i = 0; i < n; i++) {
            o = port === 0 ? applyCan(st, i === 0 ? 1 : 0, 0) : applyCan(st, 0, i === 0 ? 1 : 0);
            en = o.o1 * o.o1 + o.o2 * o.o2;
            if (!isFinite(en) || en > 1e8) return true;
            if (i < n / 2) e1 += en; else e2 += en;
          }
          if (e2 > e1 * 0.05 + 1e-18) return true;
        }
        return false;
      }
      function stabilizeCan(comp) {
        var guard = 0, names = ['S11', 'S12', 'S21', 'S22'], n, f;
        while (canGrows(comp) && guard < 16) {
          for (n = 0; n < 4; n++) {
            f = comp.filt[names[n]];
            if (f && f.a) f.a = shrinkPoles(f.a, 0.99);
          }
          guard++;
        }
        comp.stabShrink = guard;
        return comp;
      }

      // Parse a muf['tmm', kind, ...] array into a param object.
      function parseSpec(mf) {
        if (!mf || mf[0] !== 'tmm') return null;
        var kind = mf[1], IN = 0.0254, i, o = { kind: kind, rev: 0 };
        if (kind === 'triflow' || kind === 'triflow3d') {
          o.L1 = +mf[2]; o.L2 = +mf[3]; o.L3 = +mf[4];
          o.aOval = +mf[5]; o.bOval = +mf[6];
          o.dIn = +mf[7]; o.dRet = +mf[8]; o.dOut = +mf[9];
          o.sigma = +mf[10]; o.tWall = +mf[11]; o.dHole = +mf[12];
          o.Lxover = +mf[13]; o.T = +mf[14]; o.M = +mf[15];
          o.dNeck = mf[16] != null ? +mf[16] : 0.0254;
          o.rev = mf[17] != null ? +mf[17] : 0;
        } else if (kind === 'perf_through' || kind === 'perf_through3d') {
          o.L = +mf[2]; o.dShell = +mf[3]; o.dTube = +mf[4]; o.sigma = +mf[5];
          var nch = mf[6] | 0, fr = [];
          var s = 0;
          for (i = 0; i < nch; i++) { fr.push(+mf[7 + i]); s += fr[i]; }
          if (s <= 0) fr = [1];
          else for (i = 0; i < fr.length; i++) fr[i] /= s;
          o.chFrac = fr;
          o.tWall = +mf[7 + nch]; o.dHole = +mf[8 + nch]; o.T = +mf[9 + nch]; o.M = +mf[10 + nch];
          o.rev = mf[11 + nch] != null ? +mf[11 + nch] : 0;
        } else return null;
        return o;
      }

      return {
        cx: cx, tTube: tTube, tShunt: tShunt, tChain: tChain, tMul: tMul, tId: tId,
        zPerforate: zPerforate, zHelmholtz: zHelmholtz,
        tPerforatedConcentric: tPerforatedConcentric,
        tOfCan: tOfCan, tToS: tToS, tlOfS: tlOfS,
        sampleCan: sampleCan, compileCan: compileCan,
        makeState: makeState, applyCan: applyCan, applyDual: applyDual, parseSpec: parseSpec,
        tOfCan: tOfCan, chamberZ3d: chamberZ3d, tOfNet: tOfNet, tOfTriflow3d: tOfTriflow3d, tOfPerfThrough3d: tOfPerfThrough3d,
        gas: gas, skFit: skFit, fitTransfer: fitTransfer, steiglitzMcBride: steiglitzMcBride, pronyFit: pronyFit
      };
    }
    var TMM = createEngineTmm();
    var TMM_CACHE = {};
    var CAN4_PRE = /*CAN4_PRE*/{"44100|[\"tmm\",\"triflow3d\",0.21140420000000001,0.18455639999999998,0.11838939999999998,0.08255,0.19685,0.0508,0.044449999999999996,0.044449999999999996,0.11,0.0015,0.000762,0.11,621.7,0.0073,0.0254,4]":{"filt":{"S11":{"typ":"hyb","delay":4,"fir":[-0.01949048246026464,0.029476477472515087,-0.03769609449424019,0.029904579646461198,-0.006897079976775348,-0.32608078463629225,-0.21347628266368318,-0.07147634867095465,-0.12547805087675837,-0.044605965252671326,0.020845708240897758,0.014579296040925996,0.0806535374778452,0.14846207097473565,-0.06319743830361746,0.05451518661005539],"a":[1,-0.6850749509694587,1.4460202879790192,-0.7033282112832492,1.4618814300113798,-0.7658719211116565,1.2754464393084286,-0.5834126195436784,0.43782549216022004],"b":[0.21140615898901124,-0.29708219418038584,0.3141911235306946,-0.3027635283378764,0.30498730491796744,-0.2741172951276187,0.2626764783572376,-0.3592592560735426,0.08890743285527335],"tailDelay":16},"S12":{"typ":"hyb","delay":16,"fir":[0.020264531824815277,0.03703318396901257,0.003807073627708539,-0.0034230783331730746,0.03285602028026302,0.025971194866969777,-0.005369467598927237,0.008771136623053108,0.04326092630484074,0.015177047142400946,0.0031841995647098323,0.08445295294174417,0.06213733119894944,-0.018581209529031615,0.0127129850625153,0.0027072163159816715],"a":[1,-1.9728729524410382,2.841406046237017,-3.612377050658285,3.568088069568888,-2.659627604869256,1.7388327408121573,-0.9688827092880088,0.14194867465975622],"b":[-0.02428331257382236,0.061229449941883916,-0.045435167031882946,0.0951565740565757,-0.08159653987733921,0.06832859957099328,-0.09170249831295899,0.06994781364931994,-0.01699509410328309],"tailDelay":16},"S21":{"typ":"hyb","delay":16,"fir":[0.026467959934452636,0.04836987293911839,0.004972504330068291,-0.004470959455573009,0.04291398567218022,0.03392156064257278,-0.007013182170027411,0.011456178446436711,0.056504067010404245,0.019823081981911437,0.004158954533498573,0.11030589771982906,0.08115896319862793,-0.024269334895061737,0.016604715183693438,0.0035359560045474555],"a":[1,-1.9728734411855025,2.8414069553693495,-3.6123783680420223,3.5680897219947867,-2.6596291652297914,1.7388338426566317,-0.9688833911565607,0.14194901785816363],"b":[-0.03171697968869229,0.07997316933029866,-0.059343914688260536,0.12428613459590339,-0.10657508455132994,0.08924550203008368,-0.11977467094195121,0.09136041548200124,-0.022197682001329672],"tailDelay":16},"S22":{"typ":"hyb","delay":2,"fir":[-0.022584867071596893,0.0343221174071197,-0.04467482037234881,0.03968236375081845,-0.02929026735232935,-0.4294699062210492,-0.1928175605633522,-0.00988053692407735,-0.12230362243779608,-0.0019026286724158287,-0.03491954269419504,0.0631438101494744,0.059748781572985375,-0.01201627487529286,0.12812695537564406,0.03182178915397146],"a":[1,-1.6327899803048733,3.1097001658822325,-2.94754840619203,3.45520043678929,-2.4703497506458683,2.282999353219004,-1.0172774366843302,0.563867025008728],"b":[0.08716138519870692,-0.04682552572957179,0.02318951480224779,0.19384895433808175,-0.3421263658886577,0.5083889944426125,-0.4959358063148413,0.36011187905745046,-0.18299186076851925],"tailDelay":16}},"stabShrink":0},"48000|[\"tmm\",\"triflow3d\",0.21140420000000001,0.18455639999999998,0.11838939999999998,0.08255,0.19685,0.0508,0.044449999999999996,0.044449999999999996,0.11,0.0015,0.000762,0.11,621.7,0.0073,0.0254,4]":{"filt":{"S11":{"typ":"hyb","delay":5,"fir":[-0.020520083535429997,0.03059266514681131,-0.040568731258563424,0.03564300107561563,-0.023327540530624146,-0.35199468022844305,-0.16553959141633776,-0.06208667397315419,-0.14393370412952627,-0.017900545212530667,-0.031752251486694034,0.03846947654372203,0.03680007589812939,0.09563794296119905,0.12254957886381714,-0.11742101033462043],"a":[1,-1.106804141679187,1.9660413525636544,-1.6540007100624667,2.2007051457078224,-1.5736060284247075,1.557933841373312,-0.6905767580089301,0.33549988651057333],"b":[0.1283016043336977,0.030148031455419333,-0.10612163874782386,0.22164731749452377,-0.24291558897755947,0.3050312549596064,-0.28910258006222805,0.28654119742659223,-0.2878885364919477],"tailDelay":16},"S12":{"typ":"hyb","delay":18,"fir":[0.03177216176088626,0.02527949734997664,-0.004937727312026149,0.004359350862999591,0.030701697197788428,0.02492158215811631,-0.006324980545470974,0.004695222999886642,0.037641305670623314,0.023536482170817514,-0.0016203762093099667,0.040541440872740794,0.08960296954746338,0.019893403619901245,-0.01891259057158857,0.022037562043023956],"a":[1,-2.604542253495415,4.327988639914171,-5.662322152500837,5.6867733737895145,-4.350093326000918,2.666595486734103,-1.1984332862915172,0.184948039837335],"b":[-0.008923817997999715,0.006243128738408207,0.010470418367104896,0.007525477807878765,0.0150568989150567,-0.009946018769227126,0.012285910183226775,-0.040988906722864205,0.030378886971972708],"tailDelay":16},"S21":{"typ":"hyb","delay":18,"fir":[0.0414983337285045,0.03301811898772455,-0.0064492764891769545,0.0056938460251423255,0.04010017593180534,0.03255063792080497,-0.008261199079798857,0.006132536163117223,0.04916415434530388,0.030741527733312686,-0.00211640974277219,0.05295208603786566,0.11703245002117675,0.02598322105456484,-0.02470215911391163,0.028783754505174126],"a":[1,-2.604543231792552,4.3279910543484075,-5.662326062696914,5.6867782931361255,-4.350097971918571,2.6665987657376244,-1.1984351072143093,0.1849487305357041],"b":[-0.011655599017279765,0.008154309659404833,0.013675609261134335,0.009829251670641232,0.0196660182647658,-0.012990565230764618,0.016046725499614837,-0.05353639169982503,0.03967849967227222],"tailDelay":16},"S22":{"typ":"hyb","delay":1,"fir":[0.012513414712880562,-0.006960772779189181,-0.007028285507490529,0.02369700565790374,-0.048841682703952005,0.07260511622264688,-0.15051656426448587,-0.4315963516063082,-0.08032018888842282,-0.034443806594871895,-0.11563834152881348,0.010458384072726498,-0.04556343513633008,0.039854347006820307,0.08371351540821875,-0.02182968016906422],"a":[1,-1.9540957314389957,3.7831466871189363,-4.104771004045131,4.767407130542473,-3.789348688891292,3.193487776731167,-1.5440846857170845,0.7088681613964305],"b":[0.0715386746986636,-0.16231376224355268,0.3113209243303448,-0.25094244623231793,0.1962586758981358,-0.04623101382727048,0.04429188649157192,-0.025686782638522622,0.022322415509742538],"tailDelay":16}},"stabShrink":0},"44100|[\"tmm\",\"perf_through3d\",0.5715,0.13017499999999999,0.044449999999999996,0.2,3,0.22,0.45,0.33,0.0015,0.003175,519.6,0.0087,4]":{"filt":{"S11":{"typ":"hyb","delay":0,"fir":[-0.2432521145757572,-0.49931570254972446,-0.20719415184104406,0.09244620822519482,0.11765434198686926,-0.13601373319031268,0.07490165965713644,0.24379223944660408,-0.05600787503735035,-0.01563477362558716,-0.0688589004038132,-0.06925877355275076,0.08404011001786267,0.008247520649882859,0.030025716092451527,-0.05000450575855574],"a":[1,-1.0143113907720436,0.9566527448560198,-1.561311912951296,1.4294024197088373,-1.475524465853356,0.6904410419890017,-0.6946805162378233,0.7168582415045401],"b":[-0.11682768008328621,0.10919044019737703,-0.07445483683043798,0.18882820035376166,-0.19704498309960472,0.14082509939337115,-0.052530871771194435,0.04569958045798562,-0.0336572379920928],"tailDelay":16},"S12":{"typ":"hyb","delay":14,"fir":[0.017637906708513246,0.009146277412478037,0.04186364879946693,0.03585583232617448,0.022584378530747306,0.06549483143630803,0.00043582085289795507,0.04917900965296985,0.06573026966471383,-0.022210284809458684,0.09370208645506171,0.02884460057458724,-0.023203470053202908,0.11916943160636224,-0.017026234798298782,-0.0011269522469354526],"a":[1,-0.0007222847815451895,-1.1678925244502492,-1.7049465741913457,0.6871409949683167,1.3887170907646262,0.7614040945807259,-0.5667111224070888,-0.3921405714961329],"b":[0.10398784311982676,-0.06039933138229704,-0.10866160502100015,-0.06676590173923848,0.08297801380678407,0.04576139179293633,0.05555999567824535,-0.011045334718796752,-0.03910702955879736],"tailDelay":16},"S21":{"typ":"hyb","delay":14,"fir":[0.017637906708513246,0.009146277412478037,0.04186364879946693,0.03585583232617448,0.022584378530747306,0.06549483143630803,0.00043582085289795507,0.04917900965296985,0.06573026966471383,-0.022210284809458684,0.09370208645506171,0.02884460057458724,-0.023203470053202908,0.11916943160636224,-0.017026234798298782,-0.0011269522469354526],"a":[1,-0.0007222847815451895,-1.1678925244502492,-1.7049465741913457,0.6871409949683167,1.3887170907646262,0.7614040945807259,-0.5667111224070888,-0.3921405714961329],"b":[0.10398784311982676,-0.06039933138229704,-0.10866160502100015,-0.06676590173923848,0.08297801380678407,0.04576139179293633,0.05555999567824535,-0.011045334718796752,-0.03910702955879736],"tailDelay":16},"S22":{"typ":"hyb","delay":0,"fir":[-0.2314131565632115,-0.5092901803015595,-0.2036440695587462,0.09508902475611963,0.10117431207688359,-0.09823434469228846,-0.028478775357716004,0.08838245887887314,-0.049480428680891926,0.0532324526351115,0.1389046769864927,0.08692754546628848,-0.05487161818533848,-0.11096921401600446,0.019674135095283086,-0.04630243330324391],"a":[1,-1.2735777950257592,1.0324045316789738,-0.6337145932017625,0.6043735044521147,-0.8833193252749333,0.8888576785299942,-0.4680153372463355,-0.229191483400472],"b":[-0.015471026791176285,0.008389731908179845,-0.004026164431292844,0.14873817393879954,-0.18732123516516855,0.1314466801677949,-0.10692998058130193,0.028169127893692723,0.024596566755602737],"tailDelay":16}},"stabShrink":0},"48000|[\"tmm\",\"perf_through3d\",0.5715,0.13017499999999999,0.044449999999999996,0.2,3,0.22,0.45,0.33,0.0015,0.003175,519.6,0.0087,4]":{"filt":{"S11":{"typ":"hyb","delay":0,"fir":[-0.20022319414334935,-0.4997208332392921,-0.21067667635030762,0.015327461330977845,0.13458958914558852,-0.00876366871424184,-0.13644588371796487,0.24322006621798264,0.12209277051184435,-0.09300515395399869,0.034318701469993926,-0.11405050659022224,-0.03779521134036852,0.07739069114631923,-0.004843099818168125,0.04902553748503454],"a":[1,-1.03447854007999,1.1131022887217974,-1.5117474697076163,1.5426429885115795,-1.4866631331224598,0.8958978807676224,-1.074971074346503,0.6130609529072641],"b":[-0.030104605088206237,-0.03412101788679024,-0.06024542753855941,0.04844495830276082,0.02210338838394847,0.01436597782425515,-0.02145034222614481,0.015883929925228083,0.06925422477938245],"tailDelay":16},"S12":{"typ":"hyb","delay":16,"fir":[0.019594810056883315,0.014571505305126548,0.046256382717121694,0.025255967641297304,0.025264187539998793,0.06370411855460328,-0.008632416421114892,0.04730865950217493,0.07223651323521113,-0.03249692313458245,0.07453802983781749,0.06357721673214331,-0.043806401296426,0.07562062806305175,0.06366331711834268,-0.04365524917634085],"a":[1,0.6697104185198992,0.5358649019085318,-1.4398155482376294,-1.3900759955927438,-1.2167838961268538,0.6068474063688963,0.5764566399657223,0.7117131225800909],"b":[0.04845881712117933,0.08942106110894861,-0.025607311475599727,-0.10537386550914686,-0.13087231478214428,-0.04351540819100662,0.08085432781776311,0.07817720824418756,0.02172668530113018],"tailDelay":16},"S21":{"typ":"hyb","delay":16,"fir":[0.019594810056883315,0.014571505305126548,0.046256382717121694,0.025255967641297304,0.025264187539998793,0.06370411855460328,-0.008632416421114892,0.04730865950217493,0.07223651323521113,-0.03249692313458245,0.07453802983781749,0.06357721673214331,-0.043806401296426,0.07562062806305175,0.06366331711834268,-0.04365524917634085],"a":[1,0.6697104185198992,0.5358649019085318,-1.4398155482376294,-1.3900759955927438,-1.2167838961268538,0.6068474063688963,0.5764566399657223,0.7117131225800909],"b":[0.04845881712117933,0.08942106110894861,-0.025607311475599727,-0.10537386550914686,-0.13087231478214428,-0.04351540819100662,0.08085432781776311,0.07817720824418756,0.02172668530113018],"tailDelay":16},"S22":{"typ":"hyb","delay":0,"fir":[-0.20112894512014898,-0.49028711103359607,-0.22887381118668726,0.034911638226106885,0.11652431525014635,-0.0008649421406056694,-0.12259553381790621,0.06665405393900034,0.044334204897462426,-0.06450334737869831,0.09813605663619678,0.11415502500057646,0.07298773225125652,-0.021532311777949724,-0.13719154712207912,0.019587247278867383],"a":[1,-0.3892844331486882,1.5189635669478516,-0.7195791341287705,1.6490476110613914,-0.6084876564507024,1.2550698343290259,-0.2170881374020277,0.7786971145575506],"b":[-0.016086524707533204,-0.023247318519913532,0.012010828508072615,-0.07974903357293688,0.11249378652616933,-0.027199628977642203,0.09370319831719698,0.015041548471454618,-0.017824720465519348],"tailDelay":16}},"stabShrink":0},"44100|[\"tmm\",\"triflow3d\",0.21140420000000001,0.18455639999999998,0.11838939999999998,0.08255,0.19685,0.0508,0.044449999999999996,0.044449999999999996,0.11,0.0015,0.000762,0.11,701.9,0.0634,0.0254,4]":{"filt":{"S11":{"typ":"hyb","delay":3,"fir":[-0.013842330083984956,0.009926072848412857,-0.0011410271122786818,-0.020556844329273835,0.0527658049024197,-0.17777720952506026,-0.3129822564647018,-0.0921660193100938,-0.09999032632884791,-0.09838685960195304,0.038871415906133855,-0.012193052682502753,0.0905359724748949,0.14117789061133362,-0.06989132373947658,0.11386639133912357],"a":[1,-1.1038910445258974,0.8817948470734391,-0.5120712190897194,0.47844562596473394,-0.9139829515985365,0.832384779558313,-0.7690908495097616,0.1944229626710247],"b":[0.14671815417546694,-0.33374830763399094,0.26364874225546164,-0.1520993496405316,0.13601183680963108,-0.14726273386385896,0.09207496655687762,-0.1774927427218458,0.2170193832259495],"tailDelay":16},"S12":{"typ":"hyb","delay":15,"fir":[0.017832378321431856,0.03307730941888278,0.0016302438313725903,0.0011679934903822482,0.031652595770554126,0.017483511417408337,-0.007486307756133125,0.02612378591535888,0.03833868762537425,-0.006600642240303424,0.046843962730223246,0.08002782989435075,-0.0002246023553955296,0.002314938853334779,0.009494385317304063,-0.017542078515497706],"a":[1,-1.6344043805782993,2.1404405750966045,-2.7170406887274057,2.66283561283427,-1.9127610019709007,1.2889940123447252,-0.8699578600916016,0.14106804709534768],"b":[0.007311414787201473,0.028438598522515363,0.014951158998995793,-0.004373789752332647,0.020600021311793233,-0.07129835470579875,0.07578268022516639,-0.04670454405908049,0.025341729549189538],"tailDelay":16},"S21":{"typ":"hyb","delay":15,"fir":[0.023291269644319147,0.0432030163838469,0.002129298065466209,0.001525542518050247,0.041342165904397286,0.022835606749268045,-0.009778034620255532,0.034120863236387126,0.050075020571917365,-0.008621247007743253,0.061183951321107745,0.10452614516813144,-0.0002933581784758178,0.0030235936043556022,0.012400829802193095,-0.022912102550854116],"a":[1,-1.6344049808306136,2.1404414449622453,-2.717041832455924,2.6628370785594644,-1.9127623892944863,1.2889949570220836,-0.8699584932334418,0.14106846572622408],"b":[0.009549602987507823,0.037144290350242984,0.01952800526033336,-0.005712696399957291,0.0269061118988173,-0.09312434556010594,0.09898146120336118,-0.061001862771613395,0.03309939947880622],"tailDelay":16},"S22":{"typ":"hyb","delay":0,"fir":[0.017933946497769474,-0.018014852299319678,0.00910056276290696,0.00683940655252608,-0.03711132948867777,0.07823824425880235,-0.22431677740130085,-0.37043038511166443,-0.05742331082973177,-0.07248643314277958,-0.06736846505276417,-0.014289305388055788,0.030661618487214246,0.058185486425900206,0.009870220308991632,0.09928083274951666],"a":[1,-0.8373748840793799,1.895607520781549,-1.1375769751143778,1.7449696780822852,-0.9739396517684679,1.5762419381807278,-0.5437053711788993,0.6748065240545401],"b":[0.05259508898846443,0.09361251424482432,0.08191863005300287,0.060672806796780034,0.15390777071126294,-0.09701238101799015,0.22974626351702612,-0.09894519848433887,0.16720504148364856],"tailDelay":16}},"stabShrink":0},"48000|[\"tmm\",\"triflow3d\",0.21140420000000001,0.18455639999999998,0.11838939999999998,0.08255,0.19685,0.0508,0.044449999999999996,0.044449999999999996,0.11,0.0015,0.000762,0.11,701.9,0.0634,0.0254,4]":{"filt":{"S11":{"typ":"hyb","delay":4,"fir":[-0.020429912464692185,0.01885157523084218,-0.010954705884188542,-0.01318985540950221,0.05289803949207782,-0.238484842787464,-0.2677659587522015,-0.0668850163595869,-0.11612175335270508,-0.06985193712570384,-0.008571656718097374,0.02013069681690784,0.02751604129889506,0.12802407680534175,0.0674414634459213,-0.0807412024111988],"a":[1,-0.7532474497923014,1.6382957654368888,-0.9915297719211784,1.7292894481175083,-0.9869689017730299,1.3325405777140498,-0.5505915828279128,0.37601877790322064],"b":[0.1886371867224981,-0.0837175979002742,0.09431961307679627,-0.020841111067757694,0.05040668190274133,-0.011260955776196083,0.04128287680722193,-0.03963000583080088,-0.149308340802487],"tailDelay":16},"S12":{"typ":"hyb","delay":17,"fir":[0.029275220629050885,0.02027101434425217,-0.006088326523633955,0.009833185180302424,0.031755891652724785,0.010085835920595828,-0.005474645450108035,0.022731748278799143,0.03673716811830306,0.0011358109125142412,0.017779029623090692,0.08056393803083683,0.033307789101115114,-0.014238873239597384,0.01651365349437871,-0.0036941332894772382],"a":[1,-2.109475806414787,3.0763686182914296,-3.864274830933876,3.7873633783038065,-2.8227680893819915,1.817297564739612,-0.9631050203449527,0.13362780834570895],"b":[-0.014598569036749755,0.04334636258545857,-0.02708984138311039,0.06741296386102393,-0.0683894139918892,0.06403912534401492,-0.08611193802319772,0.07007469175749347,-0.020336887517366245],"tailDelay":16},"S21":{"typ":"hyb","delay":17,"fir":[0.03823702286243383,0.026476426898615086,-0.007952099949236164,0.012843343908966423,0.041477082974987436,0.013173336712614942,-0.007150557322590057,0.029690446731492768,0.04798323999125303,0.001483508130630768,0.023221589711791914,0.10522636804027675,0.043504051070844214,-0.01859771198641292,0.021568853543678258,-0.004824990418909038],"a":[1,-2.1094769715066244,3.0763709335270306,-3.864278225035671,3.787367623602674,-2.8227721061276094,1.8173004269787782,-0.9631067794567196,0.13362864488799067],"b":[-0.019067518741569756,0.05661566705371614,-0.03538267662981504,0.08804955103092836,-0.08932493933420663,0.08364288827787861,-0.11247268367999715,0.09152613689475296,-0.02656248465432683],"tailDelay":16},"S22":{"typ":"hyb","delay":2,"fir":[-0.02279548165867087,0.02426125942199384,-0.02049808615920497,0.0001567279851062875,0.036458945483678534,-0.3386038894887567,-0.2790105036401447,0.0029645561088325768,-0.11354934958184608,-0.03219322911829986,-0.01892318686142087,-0.001239905898160709,0.09549001111272123,-0.023613980375977252,0.07774200601491663,0.08815916873672018],"a":[1,-1.7760413369208818,3.3432483125746995,-3.429374376594802,3.9425430813518014,-3.0279759422889407,2.660041458700547,-1.2403414980482195,0.6407899099351618],"b":[0.016939888106419332,0.15529437080195527,-0.29346028374814986,0.5788023945098876,-0.614019084659284,0.6466081826336387,-0.4361394338180041,0.2772254290019238,-0.06717133454883435],"tailDelay":16}},"stabShrink":0},"44100|[\"tmm\",\"perf_through3d\",0.5715,0.13017499999999999,0.044449999999999996,0.2,3,0.22,0.45,0.33,0.0015,0.003175,576.1,0.075,4]":{"filt":{"S11":{"typ":"hyb","delay":0,"fir":[-0.2416640815777676,-0.478726277046943,-0.19115145364481972,0.1393858871627555,0.03623162776140782,-0.111414242690928,0.17937119206143048,0.13440213247136792,-0.0637336875553928,-0.01296498774022291,-0.08685402966507795,0.020060917729874787,0.07548615820962427,0.005919380817183384,-0.010251256210347257,-0.09578766003085423],"a":[1,-1.2220292904389545,1.6004264436786202,-0.925358100551715,0.8552425876672526,-0.8959216998955606,0.0548039434185511,0.10337260803572096,-0.5135997637947406],"b":[-0.047004602577849205,0.09865243360194882,-0.04549860399736635,0.022872373056649933,0.0009939425148552164,0.07451511497012223,-0.08322952892890381,0.049819684846424184,-0.030531391066662507],"tailDelay":16},"S12":{"typ":"hyb","delay":13,"fir":[0.013442790347902233,0.007182487941839043,0.035474847667739624,0.028939619954287406,0.020337606017254727,0.04837937642195605,-0.00043318261081541957,0.04590122919607407,0.03787193249837388,-0.0043739057091474,0.07841300715181886,-0.0045985698417212026,0.025328390543111087,0.07439247682824385,-0.031418899475114996,0.06337919030457825],"a":[1,-0.46691670351609765,-0.1485436068644212,-0.6265086266582173,0.7316668607300989,-0.7751050085311071,-0.23671397566781127,-0.09959018239925738,0.6330954641676534],"b":[0.01830657514191089,-0.04909655263979805,0.06767244372947236,-0.05442434732357367,0.0387075808724613,-0.02665562389322609,0.018120396062910216,-0.020070473059043237,0.01338527238427761],"tailDelay":16},"S21":{"typ":"hyb","delay":13,"fir":[0.013442790347902233,0.007182487941839043,0.035474847667739624,0.028939619954287406,0.020337606017254727,0.04837937642195605,-0.00043318261081541957,0.04590122919607407,0.03787193249837388,-0.0043739057091474,0.07841300715181886,-0.0045985698417212026,0.025328390543111087,0.07439247682824385,-0.031418899475114996,0.06337919030457825],"a":[1,-0.46691670351609765,-0.1485436068644212,-0.6265086266582173,0.7316668607300989,-0.7751050085311071,-0.23671397566781127,-0.09959018239925738,0.6330954641676534],"b":[0.01830657514191089,-0.04909655263979805,0.06767244372947236,-0.05442434732357367,0.0387075808724613,-0.02665562389322609,0.018120396062910216,-0.020070473059043237,0.01338527238427761],"tailDelay":16},"S22":{"typ":"hyb","delay":0,"fir":[-0.2328330369792538,-0.491193130401175,-0.1788092658608901,0.12674271317378652,0.040740191994100894,-0.09592551352070393,0.027049949420003036,0.03135553922400664,-0.024020923105742907,0.10889963489751858,0.133498141561038,-0.003091919971460249,-0.08946750662448386,-0.02373844967959588,-0.013203214028093982,-0.01766764261941989],"a":[1,-0.914327288465177,0.536194386208595,-0.1335415829559433,0.14010807480695428,-0.5119866723275199,0.6805826049911179,-0.4989424776253341,-0.2538147351701035],"b":[-0.01971510115365258,0.023371531742932692,0.10391696545666815,-0.07935894801281412,0.007764407307325549,-0.010044018512104961,-0.03394246259677676,0.04649181865387617,-0.00697308475813673],"tailDelay":16}},"stabShrink":0},"48000|[\"tmm\",\"perf_through3d\",0.5715,0.13017499999999999,0.044449999999999996,0.2,3,0.22,0.45,0.33,0.0015,0.003175,576.1,0.075,4]":{"filt":{"S11":{"typ":"hyb","delay":0,"fir":[-0.21666411120290227,-0.4711883626060381,-0.1937677298295935,0.04109451460348376,0.12824230486204322,-0.08946989708554108,-0.01751362404095041,0.26249222450623666,-0.010539732775330865,-0.0324049138454729,-0.007919646100941615,-0.1128886181530054,0.061333819819001385,0.04208558458358685,0.020983787457419835,0.007964178487154147],"a":[1,-0.5043539164581478,0.5258453585173625,-0.853746650457047,0.8321167402174016,-0.901881932398769,0.2108079677345951,-0.742016500556247,0.48031212523221634],"b":[-0.08994628901283956,-0.008331300583701486,-0.042071189750621185,0.12041419492872887,-0.04391319769362089,0.04753747528022511,-0.033551079156385714,0.12238580234303052,-0.04693528196313576],"tailDelay":16},"S12":{"typ":"hyb","delay":15,"fir":[0.016478321150638707,0.012631453022850903,0.03983851268676945,0.01888661117150664,0.023610457589379703,0.044907587112164,-0.008079046224676718,0.0498578117709703,0.03721352818644548,-0.01471386732258562,0.07598276193905042,0.013459685231183804,-0.009349165980506018,0.08527389738072981,-0.0014476617139576235,-0.001937519452448969],"a":[1,1.509672131268907,0.7148094813259804,-1.8666627204285366,-2.575475389689902,-1.342252534913375,0.8522853037569935,1.1472333555722118,0.5889928240259859],"b":[0.07663701211130036,0.08934268659666884,-0.0008730453257707642,-0.13430693465885934,-0.12294433449742911,-0.026410427105336504,0.057178092764734026,0.06038473820653894,0.01746200839076138],"tailDelay":16},"S21":{"typ":"hyb","delay":15,"fir":[0.016478321150638707,0.012631453022850903,0.03983851268676945,0.01888661117150664,0.023610457589379703,0.044907587112164,-0.008079046224676718,0.0498578117709703,0.03721352818644548,-0.01471386732258562,0.07598276193905042,0.013459685231183804,-0.009349165980506018,0.08527389738072981,-0.0014476617139576235,-0.001937519452448969],"a":[1,1.509672131268907,0.7148094813259804,-1.8666627204285366,-2.575475389689902,-1.342252534913375,0.8522853037569935,1.1472333555722118,0.5889928240259859],"b":[0.07663701211130036,0.08934268659666884,-0.0008730453257707642,-0.13430693465885934,-0.12294433449742911,-0.026410427105336504,0.057178092764734026,0.06038473820653894,0.01746200839076138],"tailDelay":16},"S22":{"typ":"hyb","delay":0,"fir":[-0.2074683498450092,-0.4755357425905274,-0.19890512204922917,0.05403071491096655,0.1035813497317249,-0.053111514059190054,-0.07560447607032886,0.09400893724348866,-0.03418016470003515,0.008021088863991358,0.13683400410762378,0.07894758959517845,0.03149017755469891,-0.10738368186171307,-0.029205523419952768,0.01440362000840604],"a":[1,-1.2120567557374162,0.8499142508042415,-0.3946096846328912,0.3870904025207456,-0.6389092105126536,0.5059075897568149,-0.016606038896488633,-0.44814749165105927],"b":[-0.05593780509594566,0.09460436631466604,-0.11661531281087041,0.1450683784117432,-0.012917489262062529,-0.0688181634878352,0.05797612672049038,-0.07617979865063125,0.05461442881856085],"tailDelay":16}},"stabShrink":0}}/*CAN4_PRE_END*/;
    var DL = 1024, DLM = DL - 1, HB = 4096, NSRC = 6;
    var TMM_RT = null; // set around kv8: array of per-can filter states
    // Q
    var Q_BORE_A = 0, Q_CRANK_R = 1, Q_ROD = 2, Q_VC = 3, Q_IVO = 4, Q_IVC = 5, Q_EVO = 6, Q_EVC = 7, Q_DV = 8, Q_LMAX = 9, Q_CAMX = 10,
      Q_CD = 11, Q_QC = 12, Q_HL = 13, Q_NP = 14, Q_BURN = 15, Q_XR0 = 16, Q_XROV = 17, Q_OVL = 18, Q_COV = 19, Q_COVR = 20, Q_MISR = 21,
      Q_ETAV = 22, Q_VD = 23, Q_VMAN = 24, Q_ATHW = 25, Q_AIDLE = 26, Q_FIDLE = 27, Q_ROL = 28, Q_J = 29, Q_F0 = 30, Q_F1 = 31, Q_F2 = 32,
      Q_FCOLD = 33, Q_STG = 34, Q_STKE = 35, Q_STRM = 36, Q_STRB = 37, Q_STIF = 38, Q_STJM = 39, Q_TEETH = 40, Q_VBAT = 41, Q_SOLID = 42,
      Q_KJET = 43, Q_VAC = 44, Q_LTH = 45, Q_RTH = 46, Q_SNM = 47, Q_SNR = 48, Q_SPARK0 = 49, Q_TAILA = 50, Q_NSEG = 51, Q_DWT = 52, Q_KNL = 53, Q_STIS = 54, Q_INSCALE = 55, Q_KMESH = 56, Q_KCOMM = 57, Q_KMOT = 58, Q_KBB = 59, Q_KAM = 60, Q_KBB2 = 61, Q_KBBHI = 62, Q_LEAD = 63, Q_TICKA = 64, Q_KVF = 65, Q_VFN = 66, Q_VFW = 67, Q_VFF = 68, Q_VFR = 69, Q_VFX = 70, Q_VWX = 71, Q_PSH = 72, Q_CYV = 73,
      Q_INTRES = 74, Q_VBANK = 75, Q_LTUBE = 76, Q_ATUBE = 77, Q_ROPEN = 78, Q_RCLOSE = 79, Q_KRES = 80, Q_F0RPM = 81, Q_F1RPM = 82, Q_FTHR = 83, Q_INTRUN = 84, Q_IRATTACH = 85, NQ = 86;
    var S_TH = 0, S_W = 1, S_DW = 2, S_TM = 3, S_PMAN = 4, S_WI = 5, S_PPL = 6, S_QTH = 7, S_PAC = 8, S_QSN = 9, S_WM = 10, S_LOCK = 11,
      S_TGAS = 12, S_MESH = 13, S_COMM = 14, S_RNG = 15, S_TPRIM = 16, S_TTAIL = 17, S_NFIRE = 18, S_NEVT = 19, S_REVS = 20, S_JET0 = 21,
      S_JET1 = 22, S_TST = 23, S_CUR = 24, S_LOAD = 25, S_TMD = 26, S_NT = 27, S_DWM = 28, S_STENV = 29, S_MOTPH = 30, S_STN = 31, S_CLUNKPH = 32, S_PREVLOCK = 33, S_STN2 = 34, S_LEADHUM = 35, S_PL = 36, S_PR = 37, S_QC = 38, NS = 39;
    var C_PHI = 0, C_M = 1, C_K = 2, C_OPEN = 3, C_PIVC = 4, C_VIVC = 5, C_TIVC = 6, C_RP = 7, C_VALID = 8, C_P = 9, C_Q = 10, C_E = 11,
      C_XRES = 12, C_ADV = 13, C_DUR = 14, C_OFF = 15, C_BANK = 16, C_PIPE = 17, C_VN = 18, C_QLP = 19, NC = 20;
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

    // v1.11.43: the per-sample kernel body is split into four helpers (verbatim code, same operation order, so the
    // output is bit-identical). One 38k-bytecode kv8 never reached JavaScriptCore's top tier (FTL; Safari / iPhone),
    // the pieces do. KX: scalar hand-over between kv8 and the helpers (no allocation).
    var KX = new Float64Array(8);
    function kArr(PF, AB, BB, AT, NPI, wi, nt, knl) {
      var p;
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
      // Flush IEEE-754 subnormals (older CPUs, including many in-car SoCs, handle them extremely slowly).
      var sa = PF[pf + F_SA], sb = PF[pf + F_SB];
      if (sa !== 0.0 && sa > -2.2250738585072014e-308 && sa < 2.2250738585072014e-308) PF[pf + F_SA] = 0.0;
      if (sb !== 0.0 && sb > -2.2250738585072014e-308 && sb < 2.2250738585072014e-308) PF[pf + F_SB] = 0.0;
    }
    }
    function kCyl(Q, S, CY, PF, AB, AT, K, NCY, th, w, rpm, pman, fs, dt, knl, wi, nt, tpr, rho_e, npol, leak, hk, vf_g, vf_w) {
      var g = G_EX, i, tg = 0.0, qin = 0.0, qinL = 0.0, qinR = 0.0, tick0 = 0.0, tick1 = 0.0;
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
        var law = Q[Q_TICKA] < 0.0 ? (0.15 + rpm / 1000.0) : Math.pow(Math.max(rpm, 1.0) / 1000.0, Q[Q_TICKA]);   // v10: flat hydraulic law (p997)
        var amp = tk * law * (Q[Q_SOLID] > 0.5 ? 1.0 : 0.35);
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
      var dq = 0.0;
      if (phi >= Q[Q_IVO] && phi < Q[Q_IVC]) {
        dq = dvd * w * Q[Q_INSCALE];
        if (Q[Q_INTRUN] <= 0.0) {   // generic intake runners (v11 int_run): the runner pipes carry the flow instead
          qin += dq;
          if (CY[c + C_BANK] < 0.5) qinL += dq; else qinR += dq;
        }
      }
      CY[c + C_QLP] = dq;
      CY[c + C_Q] = q; CY[c + C_P] = pcy;
      var pv = 0.0;
      if (Q[Q_KVF] > 0.0 && CY[c + C_OPEN] > 0.5 && aeff > 0.0 && q > 0.0) {
        // v10 exhaust-valve flow noise (v8phys.py): turbulent curtain jet, ~ rho c^2 (u/c)^n, differenced white noise
        var ce = Math.sqrt(g * R_GAS * tpr), uu = Math.min(q / aeff, ce), rn = (rnd(S) - 0.5) * 3.4641016151377544;
        pv = Q[Q_KVF] * rho_e * ce * ce * Math.pow(uu / ce, Q[Q_VFN]) * (rn - CY[c + C_VN]) * 0.5;
        if (vf_w > 0.0) {
          // short half-Hann after EVO, near-zero floor between pulses (vf_width=0 keeps v10-off identical)
          var dphi = phi - Q[Q_EVO]; if (dphi < 0.0) dphi += 720.0;
          pv *= (dphi < vf_w) ? 0.5 * (1.0 + Math.cos(Math.PI * dphi / vf_w)) : Q[Q_VFF];
        }
        if (vf_g !== 1.0) pv *= vf_g;
        CY[c + C_VN] = rn;
      }
      var q_ac = q;
      if (Q[Q_PSH] > 0.0 && CY[c + C_OPEN] > 0.5) {
        // acoustic-only: boost first 40 crank deg after EVO, ease later pumping (mass flow unchanged)
        var dphi_s = phi - Q[Q_EVO]; if (dphi_s < 0.0) dphi_s += 720.0;
        q_ac = (dphi_s < 40.0)
          ? q * (1.0 + Q[Q_PSH] * 0.5 * (1.0 + Math.cos(Math.PI * dphi_s / 40.0)))
          : q * (1.0 / (1.0 + 0.5 * Q[Q_PSH]));
      }
      if (Q[Q_CYV] > 0.0) q_ac *= 1.0 + Q[Q_CYV] * (((c / NC) | 0) % 3 - 1);
      AB[pp * DL + wi] = z0 * q_ac + rv * pin + pv;
      if (knl > 0.0) arr(AB, AT, PF, pp, wi, nt, knl);
      tg += (pcy - P_ATM) * dvd;
    }
      KX[0] = tg; KX[1] = qin; KX[2] = qinL; KX[3] = qinR; KX[4] = tick0; KX[5] = tick1;
    }
    function kNodes(NODE, NN, NW, PF, AB, BB, AT, NA, CY, S, Q, out, ko, wi, nt, knl, fs) {
      var j, qin = KX[1], qinL = KX[2], qinR = KX[3];
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
        if (!(sy > 0.0) || !isFinite(sp)) continue;
        var pj = 2.0 * sp / sy;
        if (!isFinite(pj)) continue;
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
        // tailpipe jet (flow) noise; k_jet = 0 (Sound v14+) removes it entirely: no RNG draw, no per-sample exp/sqrt
        if (Q[Q_KJET] > 0.0) {
          var uu = U / Q[Q_TAILA]; if (uu < 0.0) uu = 0.0;
          var nz = (rnd(S) - 0.5) * 3.4641016151377544;
          var fj = uu / Math.sqrt(4.0 * Q[Q_TAILA] / Math.PI) + 150.0, aj = 1.0 - Math.exp(-2.0 * Math.PI * fj / fs);
          var r4 = (uu / 30.0) * (uu / 30.0);
          S[S_JET0 + mo] += aj * (nz * 7e-4 * Q[Q_KJET] * r4 * r4 - S[S_JET0 + mo]);
          out[ko + O_JET0 + mo] = S[S_JET0 + mo];
        }
      } else if (typ === 3) {
        // closed / rigid end: pressure reflection +1, no radiation (Helmholtz cavity)
        var pc = NODE[nb + 3];
        BB[pc * DL + wi] = PF[pc * NF + F_SA];
      } else if (typ === 4) {
        // intake valve (v11 int_run): prescribed volume flow into the cylinder while IVO-IVC, closed end otherwise
        var pr4 = NODE[nb + 3], cy4 = NODE[nb + 1], z04 = RHO_AIR * PF[pr4 * NF + F_C] / PF[pr4 * NF + F_A];
        AB[pr4 * DL + wi] = PF[pr4 * NF + F_SB] - z04 * CY[cy4 * NC + C_QLP];
        if (knl > 0.0) arr(AB, AT, PF, pr4, wi, nt, knl);
      } else if (typ === 5) {
        // runner opens into a lumped plenum: p_pipe = p_plenum; U5 is flow out of the runner into the volume
        var pr5 = NODE[nb + 3], pin5 = PF[pr5 * NF + F_SA], bk5 = NODE[nb + 1], ppl5;
        if (Q[Q_INTRES] > 0.0 && bk5 !== 2) ppl5 = bk5 === 0 ? S[S_PL] : S[S_PR];
        else ppl5 = S[S_PPL];
        BB[pr5 * DL + wi] = ppl5 - pin5;
        var U5 = (2.0 * pin5 - ppl5) * PF[pr5 * NF + F_A] / (RHO_AIR * PF[pr5 * NF + F_C]);
        if (Q[Q_INTRES] > 0.0 && bk5 !== 2) { if (bk5 === 0) qinL -= U5; else qinR -= U5; }
        qin -= U5;
      } else if (typ === 7) {
        // v16.1 resistive 2-port junction (port_bc): p1 - p2 = R u, u1 + u2 = 0; r = NA[2nd+1] in narrow rho c / A
        var q1 = NODE[nb + 3], s1 = NODE[nb + 4], q2 = NODE[nb + 5], s2 = NODE[nb + 6];
        var a71 = s1 === 0 ? PF[q1 * NF + F_SB] : PF[q1 * NF + F_SA], a72 = s2 === 0 ? PF[q2 * NF + F_SB] : PF[q2 * NF + F_SA];
        var y71 = PF[q1 * NF + F_A] * PF[q1 * NF + F_C], y72 = PF[q2 * NF + F_A] * PF[q2 * NF + F_C];
        var u7 = 2.0 * (a71 - a72) / (1.0 / y71 + 1.0 / y72 + NA[nd * 2 + 1] / (y71 < y72 ? y71 : y72));
        if (!isFinite(u7)) continue;
        if (s1 === 0) { AB[q1 * DL + wi] = a71 - u7 / y71; if (knl > 0.0) arr(AB, AT, PF, q1, wi, nt, knl); } else BB[q1 * DL + wi] = a71 - u7 / y71;
        if (s2 === 0) { AB[q2 * DL + wi] = a72 + u7 / y72; if (knl > 0.0) arr(AB, AT, PF, q2, wi, nt, knl); } else BB[q2 * DL + wi] = a72 + u7 / y72;
      } else if (typ === 6) {
        // TMM two-port: travelling-wave scattering via precomputed Sij filters.
        var id6 = NODE[nb + 1], pA = NODE[nb + 3], eA = NODE[nb + 4], pB = NODE[nb + 5], eB = NODE[nb + 6];
        var pinA = eA === 0 ? PF[pA * NF + F_SB] : PF[pA * NF + F_SA];
        var pinB = eB === 0 ? PF[pB * NF + F_SB] : PF[pB * NF + F_SA];
        var yo = (TMM_RT && TMM_RT[id6]) ? (TMM_RT[id6].dual ? TMM.applyDual(TMM_RT[id6], pinA, pinB) : TMM.applyCan(TMM_RT[id6], pinA, pinB)) : { o1: 0, o2: 0 };
        if (eA === 0) { AB[pA * DL + wi] = yo.o1; if (knl > 0.0) arr(AB, AT, PF, pA, wi, nt, knl); }
        else BB[pA * DL + wi] = yo.o1;
        if (eB === 0) { AB[pB * DL + wi] = yo.o2; if (knl > 0.0) arr(AB, AT, PF, pB, wi, nt, knl); }
        else BB[pB * DL + wi] = yo.o2;
      }
    }
      KX[1] = qin; KX[2] = qinL; KX[3] = qinR;
    }
    function kStarter(Q, S, CY, out, ko, w, wm, cur, tst, G, ke, Rm, engaged, powered, dt, fs) {
    // continuous starter: mesh, motor shaft, commutator and armature noise follow w(t)/wm(t);
    // compression 'rur-rur' is the physics speed ripple plus a crank-angle AM (not one-sample spikes)
    var stOut = 0.0, stLo = 0.0, tnom = G * ke * Q[Q_VBAT] / Rm;
    if (tnom < 1e-9) tnom = 1e-9;
    if (S[S_LOCK] > 0.5 && S[S_PREVLOCK] < 0.5) S[S_STENV] += 0.85;   // pinion engagement clunk
    S[S_PREVLOCK] = S[S_LOCK];
    if (S[S_LOCK] > 0.5) {
      S[S_MESH] += Q[Q_TEETH] * w * dt / (2.0 * Math.PI);
      S[S_MESH] -= Math.floor(S[S_MESH]);
      var load = tst / tnom; if (load < 0.0) load = 0.0; else if (load > 2.5) load = 2.5;
      var aMesh = Q[Q_KMESH] * (0.40 + 0.60 * load);
      var pMesh = 6.283185307179586 * S[S_MESH];
      var mAdd = aMesh * (Math.sin(pMesh) + 0.35 * Math.sin(2.0 * pMesh) + 0.12 * Math.sin(3.0 * pMesh));
      stOut += mAdd; stLo += mAdd;
    }
    var nz = (rnd(S) - 0.5) * 3.4641016151377544;
    if (wm > 1.0) {
      S[S_MOTPH] += wm * dt / (2.0 * Math.PI);
      S[S_MOTPH] -= Math.floor(S[S_MOTPH]);
      var curN = cur / 220.0; if (curN > 1.0) curN = 1.0;
      var aMot = Q[Q_KMOT] * (0.35 + 0.65 * curN);
      var pMot = 6.283185307179586 * S[S_MOTPH];
      var motAdd = aMot * (Math.sin(pMot) + 0.35 * Math.sin(2.0 * pMot));
      stOut += motAdd; stLo += motAdd;
      S[S_COMM] += Q[Q_NSEG] * wm * dt / (2.0 * Math.PI);
      S[S_COMM] -= Math.floor(S[S_COMM]);
      var aComm = Q[Q_KCOMM] * (0.30 + 0.70 * curN);
      var pComm = 6.283185307179586 * S[S_COMM];
      stOut += aComm * (Math.sin(pComm) + 0.22 * Math.sin(2.0 * pComm)) * (1.0 + 0.28 * nz);
      var curA = 0.22 + 0.78 * (0.35 + 0.65 * curN);
      var alBb = 1.0 - Math.exp(-2.0 * Math.PI * 2000.0 / fs);
      var alBb2 = 1.0 - Math.exp(-2.0 * Math.PI * 800.0 / fs);
      S[S_STN] += alBb * (nz * Q[Q_KBB] * curA - S[S_STN]);
      S[S_STN2] += alBb2 * (nz * Q[Q_KBB2] * curA - S[S_STN2]);
      stOut += S[S_STN] + S[S_STN2] + (nz - S[S_STN]) * Q[Q_KBBHI] * curA;
    } else {
      S[S_STN] *= Math.exp(-dt / 0.010);
      S[S_STN2] *= Math.exp(-dt / 0.012);
      stOut += S[S_STN] + S[S_STN2];
    }
    var clunk = 0.0, stLow = 0.0, stHigh = 0.0;
    // stOut so far is mesh+motor+comm+broadband; split lows (mesh/motor) from the whoosh
    stLow = 0.0; stHigh = stOut;
    // re-split: mesh and motor were added first; easiest: classify by what we already mixed — apply two AMs below using stOut as the grind
    S[S_STENV] *= Math.exp(-dt / 0.020);   // solenoid / pinion body
    if (S[S_STENV] > 1e-8) {
      S[S_CLUNKPH] += 185.0 * dt;
      S[S_CLUNKPH] -= Math.floor(S[S_CLUNKPH]);
      var pCl = 6.283185307179586 * S[S_CLUNKPH];
      clunk += S[S_STENV] * (0.55 + 0.70 * Math.sin(pCl) + 0.20 * Math.sin(2.0 * pCl));
    } else S[S_STENV] = 0.0;
    // quiet low-band solenoid / engagement wash (not a held sine) while the pinion is in and the motor is not yet driving
    if (engaged && !powered) {
      S[S_LEADHUM] += (1.0 - S[S_LEADHUM]) * (dt / 0.014);
      var alLd = 1.0 - Math.exp(-2.0 * Math.PI * 350.0 / fs);
      S[S_STN] += alLd * (nz * 0.28 * S[S_LEADHUM] - S[S_STN]);
      S[S_CLUNKPH] += 155.0 * dt;
      S[S_CLUNKPH] -= Math.floor(S[S_CLUNKPH]);
      clunk += S[S_STN] + S[S_LEADHUM] * 0.07 * Math.sin(6.283185307179586 * S[S_CLUNKPH]);
    } else {
      S[S_LEADHUM] *= Math.exp(-dt / 0.020);
      if (S[S_LEADHUM] > 1e-4) clunk += S[S_LEADHUM] * 0.08 * S[S_STN];
    }
    // burst AM: long compression-to-TDC struggle, brief deep gap after TDC (whoosh dips more than the low rumble)
    var ncomp = ((CY.length / NC) | 0) * 0.5;
    var u = S[S_TH] / 360.0 * ncomp;
    u -= Math.floor(u);
    var kamH = Q[Q_KAM]; if (kamH < 0.0) kamH = 0.0; else if (kamH > 0.95) kamH = 0.95;
    var kamL = kamH * 0.42;
    // hold near TDC (struggle), brief gap after TDC, then rise through the next compression
    var rel = 0.12, gap = 0.06, hold = 0.54, rise = 1.0 - rel - gap - hold, shp;
    if (u < rel) { var x = u / rel; shp = 1.0 - x * x * (3.0 - 2.0 * x); }
    else if (u < rel + gap) shp = 0.0;
    else if (u < rel + gap + rise) { var x2 = (u - rel - gap) / rise; shp = x2 * x2; }
    else shp = 1.0;
    var amH = (1.0 - kamH) + kamH * shp, amL = (1.0 - kamL) + kamL * shp;
    // mesh/motor sit in the low half of stOut, comm+bb in the high half — a gentle tilt: lows * amL, all * mix
    stOut = stLo * amL + (stOut - stLo) * amH + clunk;
    out[ko + O_ST] += stOut;
    }

    function kIntake(Q, S, out, ko, a_th, m_th, rpm, thr, qin, qinL, qinR, dt) {
      var gp = G_AIR * P_ATM;
      if (Q[Q_INTRES] > 0.0) {
        // v10: two bank distributors joined by a resonance tube and flap (off by default)
        var vb = Q[Q_VBANK];
        S[S_PL] += gp / vb * (0.5 * S[S_QTH] - qinL - S[S_QC]) * dt;
        S[S_PR] += gp / vb * (0.5 * S[S_QTH] - qinR + S[S_QC]) * dt;
        S[S_PPL] = 0.5 * (S[S_PL] + S[S_PR]);
      } else {
        S[S_PPL] += gp / Q[Q_VMAN] * (S[S_QTH] - qin) * dt;
      }
      var ath = a_th > 2e-6 ? a_th : 2e-6, mth = RHO_AIR * Q[Q_LTH] / ath, rth = Q[Q_RTH] * RHO_AIR * C_AIR / ath;
      S[S_QTH] = (S[S_QTH] + (S[S_PAC] - S[S_PPL]) / mth * dt) / (1.0 + rth / mth * dt);
      S[S_PAC] += gp / Q[Q_VAC] * (S[S_QSN] - S[S_QTH]) * dt;
      S[S_QSN] = (S[S_QSN] - S[S_PAC] / Q[Q_SNM] * dt) / (1.0 + Q[Q_SNR] / Q[Q_SNM] * dt);
      if (Q[Q_INTRES] > 0.0) {
        var closedIR = (rpm >= Q[Q_F0RPM] && rpm <= Q[Q_F1RPM] && thr >= Q[Q_FTHR]) ? 1.0 : 0.0;
        var rcp = Q[Q_RCLOSE] * closedIR + Q[Q_ROPEN] * (1.0 - closedIR);
        var inert = RHO_AIR * Q[Q_LTUBE] / Q[Q_ATUBE];
        S[S_QC] = (S[S_QC] + (S[S_PL] - S[S_PR]) / inert * dt) / (1.0 + rcp / inert * dt);
        out[ko + O_INT] = S[S_QSN] + Q[Q_KRES] * closedIR * (S[S_PL] - S[S_PR]);
      } else {
        out[ko + O_INT] = S[S_QSN];
      }
      if (S[S_PMAN] < 0.528 * P_ATM) out[ko + O_HISS] = (rnd(S) - 0.5) * 3.4641016151377544 * m_th * Math.pow(1.0 - S[S_PMAN] / (0.528 * P_ATM), 0.5);
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
        kArr(PF, AB, BB, AT, NPI, wi, nt, knl);
        // cylinders
        tg = 0.0;
        var qin = 0.0, qinL = 0.0, qinR = 0.0, tick0 = 0.0, tick1 = 0.0, tpr = S[S_TPRIM], rho_e = P_ATM / (R_GAS * tpr);
        var npol = Q[Q_NP] - (Q[Q_NP] - 1.12) * Math.exp(-rpm / 250.0), leak = Math.exp(-dt / 3.0);
        var hk = dt / 0.03 * Math.exp(-rpm / 100.0);   // burnt-charge wall heat transfer at near-standstill (v8phys.py)
        var vf_g = 1.0, vf_w = Q[Q_VFW];
        if (rpm > Q[Q_VFR] && Q[Q_VFR] > 0.0) {
          var rat = Q[Q_VFR] / rpm;
          if (Q[Q_VFX] > 0.0) vf_g = Math.pow(rat, Q[Q_VFX]);
          if (Q[Q_VWX] > 0.0 && vf_w > 0.0) vf_w = vf_w * Math.pow(rat, Q[Q_VWX]);
        }
        kCyl(Q, S, CY, PF, AB, AT, K, NCY, th, w, rpm, pman, fs, dt, knl, wi, nt, tpr, rho_e, npol, leak, hk, vf_g, vf_w);
        tg = KX[0]; qin = KX[1]; qinL = KX[2]; qinR = KX[3]; tick0 = KX[4]; tick1 = KX[5];
        S[S_TGAS] = tg;
        // exhaust network nodes
        KX[1] = qin; KX[2] = qinL; KX[3] = qinR;
        kNodes(NODE, NN, NW, PF, AB, BB, AT, NA, CY, S, Q, out, ko, wi, nt, knl, fs);
        qin = KX[1]; qinL = KX[2]; qinR = KX[3];
        S[S_WI] = (wi + 1) & DLM;
        S[S_NT] = nt + 1.0;
        kIntake(Q, S, out, ko, a_th, m_th, rpm, thr, qin, qinL, qinR, dt);
        out[ko + O_TICK0] = tick0; out[ko + O_TICK1] = tick1;
        kStarter(Q, S, CY, out, ko, w, wm, cur, tst, G, ke, Rm, engaged, powered, dt, fs);
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
    // ---- v19 generic outside-path and road-noise helpers (set keys out_path / noise; unused by older sets) ----
    function besselJ1(x) { var s = 0.0, t = x / 2, m; for (m = 0; m < 30; m++) { s += t; t *= -(x * x / 4) / ((m + 1) * (m + 2)); if (Math.abs(t) < 1e-14) break; } return s; }
    // |R| of an unflanged pipe end: Rousseau (1996) fits to Levine & Schwinger (1948)
    function lsAbsR(ka) {
      if (ka < 1e-6) return 1.0;
      if (ka < 1.0) return Math.exp(-ka * ka / 2) * (1 + Math.pow(ka, 4) / 6 * (Math.log(1 / (1.781 * ka)) + 19 / 12));
      return Math.sqrt(Math.PI * ka) * Math.exp(-ka) * (1 + 3 / (32 * ka * ka));
    }
    // Levine & Schwinger (1948) eq. VII.5 as corrected by Ando (1968): power gain of an unflanged pipe, normalised so the
    // sphere average is 1 (an isotropic source = 1). Returns the PRESSURE factor sqrt(G). ka is clamped at 1.84 (first
    // cross mode of the pipe; the plane-wave solution is not valid above it).
    function lsGainRaw(th, ka, R) {
      var x = ka * Math.sin(th), j = Math.abs(x) < 1e-8 ? 1.0 : 2 * besselJ1(x) / x, c = Math.cos(th), s2 = 1 - c * c;
      return j * j * ((1 + c) * (1 + c) + 2 * s2 * R + (1 - c) * (1 - c) * R * R) / 4;
    }
    function lsDir(th, ka) {
      ka = Math.min(ka, 1.84); var R = lsAbsR(ka), N = 0.0, i, M = 48;
      for (i = 0; i < M; i++) { var t = (i + 0.5) * Math.PI / M; N += lsGainRaw(t, ka, R) * Math.sin(t) * Math.PI / M; }
      N /= 2;
      return Math.sqrt(lsGainRaw(th, ka, R) / N);
    }
    // ISO 9613-1 (1993) pure-tone atmospheric absorption, dB/m (T in C, RH in %, 101.325 kPa)
    function isoAlpha(f, Tc, RH) {
      var T = Tc + 273.15, T0 = 293.15, T01 = 273.16, C = -6.8346 * Math.pow(T01 / T, 1.261) + 4.6151, h = RH * Math.pow(10, C);
      var frO = 24 + 4.04e4 * h * (0.02 + h) / (0.391 + h), frN = Math.pow(T / T0, -0.5) * (9 + 280 * h * Math.exp(-4.170 * (Math.pow(T / T0, -1 / 3) - 1)));
      return 8.686 * f * f * (1.84e-11 * Math.sqrt(T / T0) + Math.pow(T / T0, -2.5) * (0.01275 * Math.exp(-2239.1 / T) / (frO + f * f / frO) + 0.1068 * Math.exp(-3352.0 / T) / (frN + f * f / frN)));
    }
    // plane-wave ground reflection, Delany & Bazley (1970) impedance (Embleton et al. 1983 form, sigma in kPa s/m2); e^{jwt}
    function dbRefl(f, sigma, sinPsi) {
      var X = f / sigma, zr = 1 + 9.08 * Math.pow(X, -0.75), zi = -11.9 * Math.pow(X, -0.73);
      var ar = zr * sinPsi - 1, ai = zi * sinPsi, br = zr * sinPsi + 1, bi = zi * sinPsi, d = br * br + bi * bi;
      return [(ar * br + ai * bi) / d, (ai * br - ar * bi) / d];
    }
    // magnitude fit of one RBJ biquad to a smooth target (dB) on a log grid by a small grid search: high shelf for direct
    // rows (directivity + incoherent ground energy + air), low-pass x gain for ground-image rows. Phase is not fitted.
    function rbjShelf(f0, gdb, fs) {
      var A = Math.pow(10, gdb / 40), w = 2 * Math.PI * f0 / fs, cw = Math.cos(w), al = Math.sin(w) / 2 * Math.sqrt(2), sa = 2 * Math.sqrt(A) * al;
      var b0 = A * ((A + 1) + (A - 1) * cw + sa), b1 = -2 * A * ((A - 1) + (A + 1) * cw), b2 = A * ((A + 1) + (A - 1) * cw - sa);
      var a0 = (A + 1) - (A - 1) * cw + sa, a1 = 2 * ((A - 1) - (A + 1) * cw), a2 = (A + 1) - (A - 1) * cw - sa;
      return [b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0];
    }
    function bqDb(b, f, fs) {
      var w = 2 * Math.PI * f / fs, c1 = Math.cos(w), s1 = Math.sin(w), c2 = Math.cos(2 * w), s2 = Math.sin(2 * w);
      var nr = b[0] + b[1] * c1 + b[2] * c2, ni = -b[1] * s1 - b[2] * s2, dr = 1 + b[3] * c1 + b[4] * c2, di = -b[3] * s1 - b[4] * s2;
      return 10 * Math.log10((nr * nr + ni * ni) / (dr * dr + di * di) + 1e-30);
    }
    function fitMag(Tdb, F, fs, kind) {
      var best = null, be = 1e30, i, j, q, fl = -40;
      var use = []; for (q = 0; q < F.length; q++) if (F[q] >= 40 && F[q] <= 12000) use.push(q);
      for (i = 0; i < 24; i++) {
        var fc = (kind === 'lp' ? 80 : 150) * Math.pow((kind === 'lp' ? 16000 / 80 : 10000 / 150), i / 23);
        var opts = kind === 'lp' ? [0.5, 0.6, 0.707] : [];
        if (kind === 'lp') {
          for (j = 0; j < opts.length; j++) {
            var b = rbj('lp', fc, opts[j], fs), off = 0, nw = 0, e = 0;
            // image rows: errors weighted by target power, so the fit follows the band where the reflection is audible
            for (q = 0; q < use.length; q++) if (Tdb[use[q]] > fl) { var wq = Math.pow(10, Tdb[use[q]] / 10); off += wq * (Tdb[use[q]] - bqDb(b, F[use[q]], fs)); nw += wq; }
            off = nw ? off / nw : 0;
            for (q = 0; q < use.length; q++) { var m = bqDb(b, F[use[q]], fs) + off, t = Tdb[use[q]]; var d = Math.max(m, fl) - Math.max(t, fl); e += Math.max(Math.pow(10, t / 10), 1e-3) * d * d; }
            if (e < be) { be = e; var g = Math.pow(10, off / 20); best = [b[0] * g, b[1] * g, b[2] * g, b[3], b[4]]; }
          }
        } else {
          for (j = 0; j < 49; j++) {
            var gd = -12 + 0.5 * j, bs = rbjShelf(fc, gd, fs), off2 = 0;
            for (q = 0; q < use.length; q++) off2 += Tdb[use[q]] - bqDb(bs, F[use[q]], fs);
            off2 /= use.length; var e2 = 0;
            for (q = 0; q < use.length; q++) { var d2 = bqDb(bs, F[use[q]], fs) + off2 - Tdb[use[q]]; e2 += d2 * d2; }
            if (e2 < be) { be = e2; var g2 = Math.pow(10, off2 / 20); best = [bs[0] * g2, bs[1] * g2, bs[2] * g2, bs[3], bs[4]]; }
          }
        }
      }
      return best;
    }
    // CNOSSOS-EU (Directive (EU) 2015/996, Appendix F, Table F-1; the 2021/1226 table is ~2.5-3.5 dB(A) louder) category 1 (light motor vehicles) rolling-noise coefficients,
    // octave bands 63 Hz - 8 kHz: L_WR = A_R + B_R log10(v / 70 km/h), dB re 1 pW
    var CN_F = [63, 125, 250, 500, 1000, 2000, 4000, 8000];
    var CN_AR = [79.7, 85.7, 84.5, 90.2, 97.3, 93.9, 84.1, 74.3], CN_BR = [30.0, 41.5, 38.9, 25.7, 32.5, 37.2, 39.0, 40.0];
    function post(n, fs, out, PV, PST, H, PATH, NPATH, BQ, BS, yL, yR, off, RQ, RS) {
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
        var x5 = bq(sv, BQ, 10, BS, 8) + 0.55 * bq(sv, BQ, 15, BS, 10);
        var st = PV[12] | 0; if (st > 0) x5 += 0.40 * bq(sv, BQ, st, BS, 64) + 1.05 * bq(sv, BQ, st + 5, BS, 66);
        H[hi] = x0; H[HB + hi] = x1; H[2 * HB + hi] = x2; H[3 * HB + hi] = x3; H[4 * HB + hi] = x4; H[5 * HB + hi] = x5;
        for (e = 0; e < 2; e++) {
          var r0 = 0.0, r1 = 0.0, r2 = 0.0, r3 = 0.0;
          for (j = 0; j < NPATH; j++) {
            var pr = j * 5;
            if ((PATH[pr + 1] | 0) !== e) continue;
            var s = PATH[pr] | 0, D = PATH[pr + 3], i0 = D | 0, fr = D - i0, hb = s * HB;
            var v = (1.0 - fr) * H[hb + ((hi - i0) & (HB - 1))] + fr * H[hb + ((hi - i0 - 1) & (HB - 1))];
            if (RQ) { var q5 = j * 5;   // v19 out_path row filter (a flat row is a gain only: a1 = a2 = 0)
              if (RQ[q5 + 3] !== 0 || RQ[q5 + 4] !== 0) { var q2 = j * 2, yq = RQ[q5] * v + RS[q2]; RS[q2] = RQ[q5 + 1] * v - RQ[q5 + 3] * yq + RS[q2 + 1]; RS[q2 + 1] = RQ[q5 + 2] * v - RQ[q5 + 4] * yq; v = yq; } else v *= RQ[q5]; }
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
      // Optional per-set intake/exhaust advertised duration and LSA (defaults: adv_dur / lsa). Unused = bit-identical.
      var dEx = ((c.ex_dur == null) ? c.adv_dur : c.ex_dur) / 2.0;
      var lsaEx = (c.ex_lsa == null) ? c.lsa : c.ex_lsa;
      var dInBase = (c.in_dur == null) ? c.adv_dur : c.in_dur;
      var lsaInBase = (c.in_lsa == null) ? c.lsa : c.in_lsa;
      var dIn = ((intakeDur == null) ? dInBase : intakeDur) / 2.0;
      var lsaIn = (intakeLsa == null) ? lsaInBase : intakeLsa;
      return [360.0 + (lsaIn - dIn), 360.0 + (lsaIn + dIn), 360.0 - (lsaEx + dEx), 360.0 - (lsaEx - dEx)];
    }
    // Generic: a parameter set may override car geometry. Keys absent from the set leave the car bit-identical.
    var SET_CAR_KEYS = ['rod','ev_d','ev_lift','camx','adv_dur','lsa','ex_dur','ex_lsa','in_dur','in_lsa','v_ac','sn_len','sn_d','th_len','v_man','x_len','x_d','pipe_d','tail_d','tail_len','inter_len','head_len','cabin_len','cabin_h','muf','helmholtz','muf_t1'];
    function applySetCar(c, P) {
      if (!P) return c;
      for (var i = 0; i < SET_CAR_KEYS.length; i++) {
        var k = SET_CAR_KEYS[i];
        if (P[k] !== undefined) {
          if (k === 'helmholtz' && (P[k] === 0 || P[k] === null)) c[k] = 0;
          else c[k] = P[k];
        }
      }
      return c;
    }
    /** Generic variable-cam block: a phase map vs rpm/load and a two-step lift switch with
     *  hysteresis thresholds, plus an optional plenum-volume flap. All numbers are parameters
     *  (defaults reproduce the 997.1 VarioCam Plus rule that was previously hardcoded). */
    function applyVariableCam(Q, c, rpm, thr, st, P) {
      st = st || {}; P = P || {};
      var idle = c.idle_rpm, red = c.redline;
      rpm = Math.max(idle * 0.5, Math.min(red * 1.05, rpm));
      thr = Math.max(0, Math.min(1, thr));
      var offRpm = P.vcam_off_rpm !== undefined ? P.vcam_off_rpm : 2500.0;
      var offThr = P.vcam_off_thr !== undefined ? P.vcam_off_thr : 0.18;
      var onRpm = P.vcam_on_rpm !== undefined ? P.vcam_on_rpm : 3000.0;
      var onThr = P.vcam_on_thr !== undefined ? P.vcam_on_thr : 0.28;
      var liftLo = P.vcam_lift_lo !== undefined ? P.vcam_lift_lo : 3.6;
      var liftHi = P.vcam_lift_hi !== undefined ? P.vcam_lift_hi : 11.0;
      var high = !!st.high;
      if (high) { if (rpm < offRpm || thr < offThr) high = false; }
      else { if (rpm > onRpm && thr > onThr) high = true; }
      st.high = high;
      Q[Q_INSCALE] = high ? 1.0 : (liftLo / liftHi);
      var phaseRpm = P.vcam_phase_rpm !== undefined ? P.vcam_phase_rpm : 4600.0;
      var load0 = P.vcam_load0 !== undefined ? P.vcam_load0 : 0.35;
      var load1 = P.vcam_load1 !== undefined ? P.vcam_load1 : 0.65;
      var loadFloor = P.vcam_load_floor !== undefined ? P.vcam_load_floor : 0.15;
      var pullRpm = P.vcam_pull_rpm !== undefined ? P.vcam_pull_rpm : 6000.0;
      var pullMin = P.vcam_pull_min !== undefined ? P.vcam_pull_min : 0.55;
      var pullSpan = P.vcam_pull_span !== undefined ? P.vcam_pull_span : 2500.0;
      var rng = P.vcam_range !== undefined ? P.vcam_range : 40.0;
      var x = (rpm - idle) / Math.max(1, phaseRpm - idle);
      if (x < 0) x = 0; if (x > 1) x = 1;
      x *= load0 + load1 * Math.max(thr, loadFloor);
      if (rpm > pullRpm) x *= Math.max(pullMin, 1 - (rpm - pullRpm) / pullSpan);
      var adv = rng * x;
      var ev = camEvents(c, c.lsa - adv);
      Q[Q_IVO] = ev[0]; Q[Q_IVC] = ev[1]; Q[Q_EVO] = ev[2]; Q[Q_EVC] = ev[3]; Q[Q_OVL] = ev[3] - ev[0];
      var flapLo = P.vcam_flap_lo !== undefined ? P.vcam_flap_lo : 3000.0;
      var flapHi = P.vcam_flap_hi !== undefined ? P.vcam_flap_hi : 5500.0;
      var flapThr = P.vcam_flap_thr !== undefined ? P.vcam_flap_thr : 0.35;
      var flapVol = P.vcam_flap_vol !== undefined ? P.vcam_flap_vol : 0.55;
      var closed = (rpm >= flapLo && rpm <= flapHi && thr >= flapThr);
      Q[Q_VMAN] = c.v_man * (closed ? flapVol : 1);
      st.flap_closed = closed; st.advance = adv;
      return st;
    }
    function applyVariocamPlus(Q, c, rpm, thr, st, P) { return applyVariableCam(Q, c, rpm, thr, st, P); }
    function mufAsCavity(c, corePack) {
      // generic topology: series muffler chambers (after the first, the catalyst) become equal-volume side-branch cavities
      corePack = corePack || 0.0;
      var out = [], first = true, IN = 0.0254, i, item, kind, L, ratio, pack, dCav;
      for (i = 0; i < c.muf.length; i++) {
        item = c.muf[i]; kind = item[0];
        if (kind === 'chamber' && first) { out.push(item); first = false; continue; }
        if (kind === 'chamber') {
          L = item[1]; ratio = item[2]; pack = item[3];
          dCav = c.pipe_d * Math.sqrt(ratio);
          out.push(['cavity', L, dCav, pack, 0.05, 1.5 * IN, L, corePack]);
        } else out.push(item);
      }
      return out;
    }
    function perspective(c, name) {
      var m = c.mouth[0], x, z;
      if (name === 'rear') { var mm = [m[0] - 1.0, m[1], 0.5]; return { ears: [[mm[0], mm[1] + 0.09, mm[2]], [mm[0], mm[1] - 0.09, mm[2]]], ground: 0.8, outside: true, shadow: 0.3 }; }
      if (name === 'rear_far') return { ears: [[-4.0, 0.09, 1.5], [-4.0, -0.09, 1.5]], ground: 0.8, outside: true, shadow: 0.3 };
      if (name === 'engine_bay') { x = c.intake[0]; z = c.intake[2]; return { ears: [[x - 0.25, 0.09, z + 0.55], [x - 0.25, -0.09, z + 0.55]], ground: 0.8, outside: true, shadow: 0.5, bay: true }; }
      var h = c.head; return { ears: [[h[0], h[1] + 0.09, h[2]], [h[0], h[1] - 0.09, h[2]]], ground: 0.0, outside: false, shadow: 1.0, window: name === 'in_cabin_open' };
    }
    /** exhaust network (v8phys.build_net / net_arrays).
     *  fs is the worklet sample rate: every waveguide is at least one travel sample long at a
     *  conservative hot-gas speed (CFL). Shorter building-block stubs (1 mm perf end-caps,
     *  ~4 mm perforate necks) would otherwise be clamped to a 1-sample delay line — a stiff
     *  closed-end loop whose length changes with sample rate and gas temperature. */
    // Soft cap on waveguide count. Networks under this stay bit-identical; heavier
    // perforated-can sets (Cadillac p3–p7: 80–88 pipes) are rebuilt with a cheaper
    // generic can (1 segment, no 1 mm CFL-clamped end-cap stubs) so a slower CPU
    // (Tesla in-car) can finish a worklet quantum. 911 p2 is 46 pipes and is unchanged.
    var NET_PIPE_BUDGET = 56;
    function buildNet(c, fs, lim) {
      var pipes = [], nodes = [], left = {}, xp = -1, tmmCans = [];
      fs = fs > 0 ? fs : 44100;
      lim = lim || {};
      var perfNsegMax = lim.perfNsegMax, perfCaps = lim.perfCaps !== false;
      // CFL length: one sample at a conservative hot exhaust speed (900 K) or at ambient for intake (grp 3).
      var L_CFL_EX = Math.sqrt(G_EX * R_GAS * 900.0) / fs, L_CFL_IN = C_AIR / fs;
      c.left_bank.forEach(function (x) { left[x] = 1; });
      function pipe(L, d, grp, pack) {
        var Lmin = grp === 3 ? L_CFL_IN : L_CFL_EX;
        if (L < Lmin) L = Lmin;
        pipes.push([L, d, Math.PI * d * d / 4, grp, pack || 0, 0]);
        return pipes.length - 1;
      }
      function node(t, x, ports) { nodes.push([t, x, ports]); }
      for (var bank = 0; bank < 2; bank++) {
        var cyls = [], i, head, prev, dl = bank === 1 ? c.bank_dl : 0.0, pendingTmm = null;
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
        var mufItems = c.muf.slice(), prevOut = head;
        // cat_early: first chamber (cascade cat) just after head_len; total path length unchanged. Off by default.
        if (c.cat_early && mufItems.length && mufItems[0][0] === 'chamber') {
          var ce = mufItems.shift(), cd = c.pipe_d * Math.sqrt(ce[2]), catp = pipe(ce[1], cd, 2, ce[3]);
          node(0, 0, [[head, 1], [catp, 0]]);
          prevOut = catp;
        }
        var inter = pipe(c.inter_len + dl, c.pipe_d, 1);
        node(0, 0, hx ? [[prevOut, 1], [xp, bank], [inter, 0]] : [[prevOut, 1], [inter, 0]]);
        prev = inter;
        var t1Done = false;
        mufItems.forEach(function (mf) {
          if (mf[0] === 'tmm') {
            tmmCans.push(mf);
            pendingTmm = tmmCans.length - 1;
          } else if (pendingTmm !== null && (mf[0] === 'pipe' || mf[0] === 'chamber')) {
            var dT = mf[0] === 'chamber' ? c.pipe_d * Math.sqrt(mf[2]) : mf[2];
            var pT = pipe(mf[1], dT, 2, mf[3]);
            node(6, pendingTmm, [[prev, 1], [pT, 0]]);
            prev = pT;
            pendingTmm = null;
          } else if (mf[0] === 'cavity') {
            var Ln = mf.length > 4 ? mf[4] : 0.05, dn = mf.length > 5 ? mf[5] : 0.0381, Lm = mf.length > 6 ? mf[6] : 0.12;
            var main = pipe(Lm, c.pipe_d, 2, 0.0), neck = pipe(Ln, dn, 2, 0.0), cav = pipe(mf[1], mf[2], 2, mf[3]);
            node(0, 0, [[prev, 1], [main, 0], [neck, 0]]);
            node(0, 0, [[neck, 1], [cav, 0]]);
            if (mf.length > 9 && +mf[8] > 0) {   // v1.11.40 generic: absorbent layer on the closed end (thickness mf[8], packing mf[9])
              var lay = pipe(+mf[8], mf[2], 2, +mf[9]); node(0, 0, [[cav, 1], [lay, 0]]); node(3, 0, [[lay, 1]]);
            } else node(3, 0, [[cav, 1]]);
            prev = main;
          } else if (mf[0] === 'perf') {
            // perforated straight-through (same topology as v8phys.build_net)
            var Lp = +mf[1], dp = +mf[2], Ls = +mf[3], ds = +mf[4], por = +mf[5], pk = +mf[6];
            var nseg = mf.length > 7 ? (mf[7] | 0) : 4, tWall = mf.length > 8 ? +mf[8] : 0.0015, dHole = mf.length > 9 ? +mf[9] : 0.004;
            var packN = mf.length > 10 ? +mf[10] : 0.0;
            if (perfNsegMax !== undefined && nseg > perfNsegMax) nseg = perfNsegMax;
            if (nseg < 1) nseg = 1;
            var Lt = Lp / nseg, Lc = Ls / nseg, dAnn = Math.sqrt(Math.max(ds * ds - dp * dp, 1e-8));
            var sig = Math.min(Math.max(por, 1e-6), 0.9), Ah = sig * Math.PI * dp * Lt;
            var dNeck = 2.0 * Math.sqrt(Ah / Math.PI), Lneck = tWall + 0.85 * dHole;
            var Ts = [], Cs = [], Ns = [], ii;
            for (ii = 0; ii < nseg; ii++) { Ts.push(pipe(Lt, dp, 2, 0.0)); Cs.push(pipe(Lc, dAnn, 2, pk)); Ns.push(pipe(Lneck, dNeck, 2, packN)); }
            if (perfCaps) {
              var capL = pipe(0.001, dAnn, 2, pk), capR = pipe(0.001, dAnn, 2, pk);
              node(3, 0, [[capL, 1]]);
              node(0, 0, [[capL, 0], [Cs[0], 0], [Ns[0], 1]]);
              for (ii = 0; ii < nseg; ii++) {
                node(0, 0, [[prev, 1], [Ts[ii], 0], [Ns[ii], 0]]);
                prev = Ts[ii];
                if (ii + 1 < nseg) node(0, 0, [[Cs[ii], 1], [Cs[ii + 1], 0], [Ns[ii + 1], 1]]);
              }
              node(0, 0, [[Cs[nseg - 1], 1], [capR, 0]]);
              node(3, 0, [[capR, 1]]);
            } else {
              // Cheaper generic can: 1-segment (or reduced-nseg) through + neck + annulus, rigid ends,
              // no 1 mm CFL-clamped cap stubs. The neck joins the left end of the annulus.
              node(0, 0, [[Cs[0], 0], [Ns[0], 1]]);
              for (ii = 0; ii < nseg; ii++) {
                node(0, 0, [[prev, 1], [Ts[ii], 0], [Ns[ii], 0]]);
                prev = Ts[ii];
                if (ii + 1 < nseg) node(0, 0, [[Cs[ii], 1], [Cs[ii + 1], 0], [Ns[ii + 1], 1]]);
              }
              node(3, 0, [[Cs[nseg - 1], 1]]);
            }
          } else if (mf[0] === 'par') {
            // v11 generic (v1.11.39): N parallel paths (internal tubes, a perforated baffle's holes as one lumped
            // neck, a small equalising tube) from the previous volume into the next chamber; same as v8phys.build_net.
            var paths = mf[1], ch = pipe(+mf[2], c.pipe_d * Math.sqrt(+mf[3]), 2, +mf[4]), ps = [], pi_;
            for (pi_ = 0; pi_ < paths.length; pi_++) ps.push(pipe(+paths[pi_][0], +paths[pi_][1], 2, paths[pi_].length > 2 ? +paths[pi_][2] : 0.0));
            node(0, 0, [[prev, 1]].concat(ps.map(function (q) { return [q, 0]; })));
            node(0, 0, ps.map(function (q) { return [q, 1]; }).concat([[ch, 0]]));
            prev = ch;
          } else if (mf[0] === 'chx') {
            // v16.1 generic: plain chamber with extended inlet / outlet tubes (Munjal 2014 ch. 3; Selamet & Ji 1998):
            // ['chx', L, area ratio, pack, l_in, l_out, d_in, d_out]. Each extension is a tube of the pipe diameter
            // reaching l into the shell; the annulus between tube and shell is a quarter-wave cavity closed by the
            // end plate. Same gas-temperature rule as 'chamber' (muf_t1).
            var gX = (c.muf_t1 && !t1Done) ? 1 : 2; if (c.muf_t1) t1Done = true;
            var dX = c.pipe_d * Math.sqrt(mf[2]), lIn = +(mf[4] || 0), lOut = +(mf[5] || 0);
            var dIn = mf[6] ? +mf[6] : c.pipe_d, dOut = mf[7] ? +mf[7] : c.pipe_d;
            if (lIn > 0) {
              var eIn = pipe(lIn, dIn, gX, 0), aIn = pipe(lIn, Math.sqrt(Math.max(dX * dX - dIn * dIn, 1e-8)), gX, mf[3]);
              node(0, 0, [[prev, 1], [eIn, 0]]);
              var mX = pipe(Math.max(mf[1] - lIn - lOut, 0.01), dX, gX, mf[3]);
              node(0, 0, [[eIn, 1], [aIn, 0], [mX, 0]]); node(3, 0, [[aIn, 1]]); prev = mX;
            } else { var mX0 = pipe(Math.max(mf[1] - lOut, 0.01), dX, gX, mf[3]); node(0, 0, [[prev, 1], [mX0, 0]]); prev = mX0; }
            if (lOut > 0) {
              var aOut = pipe(lOut, Math.sqrt(Math.max(dX * dX - dOut * dOut, 1e-8)), gX, mf[3]), eOut = pipe(lOut, dOut, gX, 0);
              node(0, 0, [[prev, 1], [aOut, 0], [eOut, 0]]); node(3, 0, [[aOut, 1]]); prev = eOut;
            }
          } else {
            var d = mf[0] === 'chamber' ? c.pipe_d * Math.sqrt(mf[2]) : mf[2];
            // v16 generic (set key muf_t1): plain items up to and including the first chamber take the group-1 gas
            // temperature, as a first TMM can fed by the inter pipe does (tools/can_gas.cjs); later items stay at T_tail.
            // Absent key: group 2 as before (older sets bit-identical).
            var gT = (c.muf_t1 && !t1Done) ? 1 : 2;
            if (c.muf_t1 && mf[0] === 'chamber') t1Done = true;
            var p = pipe(mf[1], d, gT, mf[3]); node(0, 0, [[prev, 1], [p, 0]]); prev = p;
          }
        });
        var tail = pipe(c.tail_len, (c.tail_d != null ? c.tail_d : c.pipe_d), 2);
        if (pendingTmm !== null) {
          node(6, pendingTmm, [[prev, 1], [tail, 0]]);
          pendingTmm = null;
        } else if (c.helmholtz) {
          var hz = c.helmholtz, neck = pipe(hz[0], hz[1], 2, 0.0), cav = pipe(hz[2], hz[3], 2, 0.1);
          node(0, 0, [[prev, 1], [tail, 0], [neck, 0]]);
          node(0, 0, [[neck, 1], [cav, 0]]);
          node(3, 0, [[cav, 1]]);
        } else {
          node(0, 0, [[prev, 1], [tail, 0]]);
        }
        node(2, c.cross_mouth ? 1 - bank : bank, [[tail, 1]]);   // 997.1: each bank's cat pipe crosses to the other side
      }
      if (c.int_run) {
        // generic intake runners (v11): one 1D pipe (or nseg chain) per cylinder, intake valve (4) to bank / single plenum (5)
        var Lr = c.int_run_l, dr = c.int_run_d, nsg = Math.max(1, c.int_run_nseg | 0), att = c.int_run_attach | 0;
        for (var ci = 0; ci < c.fire_order.length; ci++) {
          var pv_ = -1;
          for (var sg = 0; sg < nsg; sg++) {
            var rp_ = pipe(Lr / nsg, dr, 3, 0.0);
            if (sg === 0) node(4, ci, [[rp_, 0]]); else node(0, 0, [[pv_, 1], [rp_, 0]]);
            pv_ = rp_;
          }
          node(5, att ? (left[ci + 1] === 1 ? 0 : 1) : 2, [[pv_, 1]]);
        }
      }
      // v16.1 generic (set key port_bc): every 2-port exhaust junction with an area step >= 1.2 becomes a resistive
      // junction (type 7) carrying the linearised mean-flow pressure loss Dp = K rho U u (Borda-Carnot; Ronneberger
      // 1967; Munjal 2014 ch. 2): expansion K = (1 - A_up/A_dn)^2, contraction K = 0.5 (1 - A_dn/A_up) (Idelchik),
      // both on the narrow pipe's velocity. k7 = [node, K, narrow pipe]; r = K M is set per block in _pipes.
      var k7 = [];
      if (c.port_bc) nodes.forEach(function (n, k) {
        if (n[0] !== 0 || n[2].length !== 2) return;
        var pu = n[2][0][0], pd = n[2][1][0], Au = pipes[pu][2], Ad = pipes[pd][2];
        if (pipes[pu][3] === 0 || pipes[pu][3] === 3 || pipes[pd][3] === 3) return;
        if (Math.max(Au, Ad) < 1.2 * Math.min(Au, Ad)) return;
        var K = Au < Ad ? (1 - Au / Ad) * (1 - Au / Ad) : 0.5 * (1 - Ad / Au);
        n[0] = 7; k7.push([k, K, Au < Ad ? pu : pd]);
      });
      var NW = 0; nodes.forEach(function (n) { NW = Math.max(NW, 3 + 2 * n[2].length); });
      var NODE = new Int32Array(nodes.length * NW);
      nodes.forEach(function (n, k) {
        NODE[k * NW] = n[0]; NODE[k * NW + 1] = n[1]; NODE[k * NW + 2] = n[2].length;
        n[2].forEach(function (pe, j) { NODE[k * NW + 3 + 2 * j] = pe[0]; NODE[k * NW + 4 + 2 * j] = pe[1]; });
      });
      return { PG: pipes, NODE: NODE, NN: nodes.length, NW: NW, tmmCans: tmmCans, k7: k7 };
    }

    /** the synth: v8phys.V8Phys + Ctl. spec = { car_key, idle_rpm, redline, name }, listener = { position } */
    function V8pSynth(spec, listener, fs, opts) {
      opts = opts || {};
      var ENGINE_KEYS = { CADDY: 'cadillac59', P997: 'p997', GT350: 'gt350' };
      var key = spec.car_key || ENGINE_KEYS[spec.engineKey] || 'gt350';
      var D = DATA[key];
      if (!D) throw new Error('v8p: no data for ' + key);
      var variant = spec.variant || opts.variant;
      var pack = (variant && D.sets && D.sets[variant]) ? D.sets[variant] : D.phys;
      this.key = key; this.c = Object.assign({}, D.car); this.P = Object.assign({}, pack, opts.phys || {}); this.fs = fs;
      applySetCar(this.c, this.P);
      this.spec = { idle_rpm: this.c.idle_rpm, redline: this.c.redline, name: this.c.name, car_key: key };
      this.vcam = !!(this.P.vcam || spec.variocam_plus || opts.variocam_plus);
      this.variocamPlus = this.vcam;
      this._vc = {};
      this.selfGated = opts.mode !== 'run' && opts.mode !== 'raw';
      var netCar = Object.assign({}, this.c);
      if (this.P.cat_early) netCar.cat_early = 1;
      if (this.P.port_bc) netCar.port_bc = 1;
      if (this.P.muf) netCar.muf = this.P.muf;   // generic: any car may select a muffler list via parameters
      else if (this.P.muf_cavity) netCar.muf = mufAsCavity(netCar, this.P.muf_core_pack || 0.0);
      if (this.P.int_run) {   // generic intake-runner block (parameters only; off unless a set enables it)
        netCar.int_run = 1;
        netCar.int_run_l = this.P.int_run_l !== undefined ? this.P.int_run_l : 0.216;
        netCar.int_run_d = this.P.int_run_d !== undefined ? this.P.int_run_d : 0.045;
        netCar.int_run_nseg = this.P.int_run_nseg !== undefined ? this.P.int_run_nseg : 1;
        netCar.int_run_attach = this.P.int_run_attach !== undefined ? this.P.int_run_attach : 1;
      }
      var N = buildNet(netCar, fs), nPipes0 = N.PG.length;
      if (nPipes0 > NET_PIPE_BUDGET) N = buildNet(netCar, fs, { perfNsegMax: 1, perfCaps: false });
      this.PG = N.PG; this.NODE = N.NODE; this.NN = N.NN; this.NW = N.NW; this.NP = N.PG.length;
      this.K7 = N.k7 && N.k7.length ? N.k7 : null;
      this.NP0 = nPipes0; this.netLite = nPipes0 > NET_PIPE_BUDGET; this.nanResets = 0;
      this.tmmCans = []; this.tmmState = [];
      if (N.tmmCans && N.tmmCans.length) {
        var ti;
        for (ti = 0; ti < N.tmmCans.length; ti++) {
          var tsp = TMM.parseSpec(N.tmmCans[ti]);
          if (!tsp) continue;
          // v14.1: compile each distinct can once per sample rate (both banks share one spec); the compiled filters are
          // read-only at run time, every can keeps its own state. Same numbers as compiling twice.
          var tkey = fs + '|' + JSON.stringify(N.tmmCans[ti]);
          var tcomp = TMM_CACHE[tkey];
          if (!tcomp && tsp.rev >= 4 && CAN4_PRE[tkey]) {
            // prebuilt rev-4 filters (tools/prebuild_cans.cjs, same compileCan4 output to the last bit) skip the
            // load-time compile at 44.1 / 48 kHz; any other rate compiles below
            var pre = CAN4_PRE[tkey];
            tcomp = { kind: tsp.kind, p: tsp, fs: fs, fsFit: fs / 4, decim: 4, filt: pre.filt, stabShrink: pre.stabShrink, prebuilt: true };
            TMM_CACHE[tkey] = tcomp;
          }
          if (!tcomp) tcomp = TMM_CACHE[tkey] = TMM.compileCan(tsp.kind, tsp, fs);
          this.tmmCans.push(tcomp);
          this.tmmState.push(TMM.makeState(tcomp));
        }
      }
      // v15 (generic param muf_cruise): a second gas state per TMM can, same geometry, compiled / prebuilt alongside.
      this.tmmDual = null;
      if (this.P.muf_cruise && this.P.muf && this.tmmCans.length) {
        var mufS = this.P.muf.map(function (x) { return JSON.stringify(x); }), dual = [], di, ordn = {}, oc = 0, mi;
        for (mi = 0; mi < this.P.muf.length; mi++) if (this.P.muf[mi][0] === 'tmm') ordn[mi] = oc++;
        for (di = 0; di < N.tmmCans.length; di++) {
          var mIdx = mufS.indexOf(JSON.stringify(N.tmmCans[di])), alt = mIdx >= 0 ? this.P.muf_cruise[mIdx] : null, asp = alt ? TMM.parseSpec(alt) : null;
          if (!asp) { dual.push(null); continue; }
          var akey = fs + '|' + JSON.stringify(alt), acomp = TMM_CACHE[akey];
          if (!acomp && CAN4_PRE[akey]) acomp = TMM_CACHE[akey] = { kind: asp.kind, p: asp, fs: fs, fsFit: fs / 4, decim: 4, filt: CAN4_PRE[akey].filt, stabShrink: CAN4_PRE[akey].stabShrink, prebuilt: true };
          if (!acomp) acomp = TMM_CACHE[akey] = TMM.compileCan(asp.kind, asp, fs);
          var isp = this.tmmCans[di].p, tri = isp.kind === 'triflow' || isp.kind === 'triflow3d';
          // live Mach rule = tools/can_gas.cjs: first can in the line fed at the group-1 temperature, later cans at T_tail;
          // switch at the geometric mean of the two design Mach numbers (nearest design state in log M; the grazing-flow
          // resistance of the perforates is proportional to M), +-20 % hysteresis
          dual.push({ comps: [this.tmmCans[di], acomp], first: ordn[mIdx] === 0, area: Math.PI * Math.pow(tri ? isp.dIn : isp.dTube, 2) / 4,
            mMid: Math.sqrt(Math.max(isp.M, 1e-6) * Math.max(asp.M, 1e-6)) });
        }
        this.tmmDual = dual;
        this.tmmState = this._tmmStates();
      }
      // gas_dyn: exhaust volume downstream of the ports (pipes in groups 1/2 + TMM can shells), for the gas transport time.
      var vex = 0.0, gi, tsp2;
      for (gi = 0; gi < this.PG.length; gi++) if (this.PG[gi][3] === 1 || this.PG[gi][3] === 2) vex += this.PG[gi][0] * this.PG[gi][2];
      for (gi = 0; gi < this.tmmCans.length; gi++) {
        tsp2 = this.tmmCans[gi].p;
        vex += (tsp2.kind === 'triflow' || tsp2.kind === 'triflow3d') ? Math.PI / 4 * tsp2.aOval * tsp2.bOval * (tsp2.L1 + tsp2.L2 + tsp2.L3) : Math.PI / 4 * tsp2.dShell * tsp2.dShell * tsp2.L;
      }
      this._vex = vex;
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
      Q[Q_TICKA] = P.tick_alpha === undefined ? -1.0 : P.tick_alpha; Q[Q_KVF] = P.k_vf || 0.0; Q[Q_VFN] = P.vf_n === undefined ? 2.0 : P.vf_n; Q[Q_VFW] = P.vf_width || 0.0; Q[Q_VFF] = P.vf_floor || 0.0; Q[Q_VFR] = P.vf_rpm_ref === undefined ? 4500.0 : P.vf_rpm_ref; Q[Q_VFX] = P.vf_rpm_exp || 0.0; Q[Q_VWX] = P.vf_w_exp || 0.0; Q[Q_PSH] = P.k_psharp || 0.0; Q[Q_CYV] = P.k_cyl || 0.0;
      if (P.int_res) {
        // two-bank resonance tube (v10 int_res): off by default. Tube L/d are estimates.
        var Lt = P.int_tube_l !== undefined ? P.int_tube_l : 0.30;
        var dtube = P.int_tube_d !== undefined ? P.int_tube_d : 0.045;
        var At = Math.PI * dtube * dtube / 4;
        Q[Q_INTRES] = 1.0; Q[Q_VBANK] = c.v_man / 2.0; Q[Q_LTUBE] = Lt; Q[Q_ATUBE] = At;
        var z0 = RHO_AIR * C_AIR / At;
        Q[Q_ROPEN] = 0.15 * z0; Q[Q_RCLOSE] = 12.0 * z0;
        Q[Q_KRES] = P.int_kres !== undefined ? P.int_kres : 3.0e-7;
        Q[Q_F0RPM] = P.int_flap_lo !== undefined ? P.int_flap_lo : 2600.0;
        Q[Q_F1RPM] = P.int_flap_hi !== undefined ? P.int_flap_hi : 5100.0;
        Q[Q_FTHR] = P.int_flap_thr !== undefined ? P.int_flap_thr : 0.30;
      }
      if (P.int_run) { Q[Q_INTRUN] = 1.0; Q[Q_IRATTACH] = P.int_run_attach !== undefined ? (P.int_run_attach | 0) : 1; }
      Q[Q_KJET] = P.k_jet; Q[Q_VAC] = c.v_ac; Q[Q_LTH] = c.th_len; Q[Q_RTH] = P.r_th;
      var asn = Math.PI * c.sn_d * c.sn_d / 4;
      Q[Q_SNM] = RHO_AIR * (c.sn_len + 0.85 * c.sn_d) / asn;
      // r_ac: generic air-cleaner element flow resistance (Pa·s/m³), default 0.
      // Added in series with the existing 0.1 ρc/A snorkel radiation term.
      // Only applied when set, so cars that omit it stay bit-identical.
      Q[Q_SNR] = 0.1 * RHO_AIR * C_AIR / asn;
      if (P.r_ac) Q[Q_SNR] += P.r_ac;
      Q[Q_SPARK0] = c.spark0; Q[Q_TAILA] = Math.PI * c.pipe_d * c.pipe_d / 4; Q[Q_DWT] = P.dw_tau; Q[Q_KNL] = P.k_nl;
      Q[Q_INSCALE] = 1.0;
      Q[Q_KMESH] = P.k_st_mesh !== undefined ? P.k_st_mesh : 0.6;
      Q[Q_KCOMM] = P.k_st_comm !== undefined ? P.k_st_comm : 0.45;
      Q[Q_KMOT] = P.k_st_mot !== undefined ? P.k_st_mot : 0.4;
      Q[Q_KBB] = P.k_st_bb !== undefined ? P.k_st_bb : 0.55;
      Q[Q_KAM] = P.k_st_am !== undefined ? P.k_st_am : 0.55;
      Q[Q_KBB2] = P.k_st_bb2 !== undefined ? P.k_st_bb2 : 0.5;
      Q[Q_KBBHI] = P.k_st_bbhi !== undefined ? P.k_st_bbhi : 0.15;
      Q[Q_LEAD] = P.st_lead !== undefined ? P.st_lead : 0.04;
      return Q;
    };
    V8pSynth.prototype._pipes = function () {
      var P = this.P, fs = this.fs, S = this.S, tp = S[S_TPRIM], tt = S[S_TTAIL], tm = T_AMB + 0.65 * (tp - T_AMB), p, nd;
      var tail = {};
      var runPl = {};
      for (nd = 0; nd < this.NN; nd++) if (this.NODE[nd * this.NW] === 2) tail[this.NODE[nd * this.NW + 3]] = 1;
      for (nd = 0; nd < this.NN; nd++) if (this.NODE[nd * this.NW] === 5) runPl[this.NODE[nd * this.NW + 3]] = 1;
      var TAX = null;
      if (P.t_axial && this.Q) {
        // v17 generic (set key t_axial = outside heat-transfer coefficient h_o, W/m2K): axial gas-temperature drop along each
        // exhaust run (consecutive group-1/2 pipes in network order, i.e. one bank from the head pipe to the tail). Per pipe,
        // steady 1-D heat balance (Incropera & DeWitt ch. 8, constant outside temperature): T_out = T_amb + (T_in - T_amb)
        // exp(-U P L / (m cp)), U = 1 / (1/h_i + 1/h_o), h_i from Dittus-Boelter (1930) Nu = 0.023 Re^0.8 Pr^0.3 (laminar
        // floor Nu 3.66), P = pi d. The pipe's acoustic temperature is the value at its mid-length. The run starts at the model's head-pipe temperature tm.
        // Mass flow per bank = the model's own exhaust flow (same formula as port_bc / can_gas). Replaces tm / T_tail only
        // for those pipes; everything else unchanged.
        TAX = new Float64Array(this.NP);
        var rpmA = S[S_W] * 60 / (2 * Math.PI), prA = S[S_PMAN] / P_ATM; if (!(rpmA > 0)) rpmA = 0; if (!(prA > 0.05)) prA = 0.05;
        var mdot = Math.max(1e-4, 0.5 * P.eta_v * prA * this.Q[Q_VD] * rpmA / 120 * P_ATM / (R_GAS * T_AMB)), cpA = G_EX * R_GAS / (G_EX - 1), Tin = tm;
        for (p = 0; p < this.NP; p++) {
          var gA = this.PG[p];
          if (gA[3] !== 1 && gA[3] !== 2) { Tin = tm; continue; }
          var dA = gA[1], TmA = Tin, mu = 1.458e-6 * TmA * Math.sqrt(TmA) / (TmA + 110.4);
          var Re = 4 * mdot / (Math.PI * dA * mu), Nu = Re > 2300 ? 0.023 * Math.pow(Re, 0.8) * Math.pow(0.7, 0.3) : 3.66;
          var hi = Nu * mu * cpA / 0.7 / dA, U = 1 / (1 / hi + 1 / P.t_axial), ex = U * Math.PI * dA * gA[0] / (mdot * cpA);
          TAX[p] = T_AMB + (Tin - T_AMB) * Math.exp(-0.5 * ex); Tin = T_AMB + (Tin - T_AMB) * Math.exp(-ex);
        }
      }
      for (p = 0; p < this.NP; p++) {
        var g = this.PG[p], L = g[0], d = g[1], A = g[2], grp = g[3], pack = g[4];
        var T = grp === 0 ? tp : (grp === 1 ? tm : tt);
        if (TAX && (grp === 1 || grp === 2)) T = TAX[p];
        var c = grp === 3 ? C_AIR : Math.sqrt(G_EX * R_GAS * T);   // grp 3: intake runners, ambient air
        var Le = L + (tail[p] ? 0.61 * d / 2 : (runPl[p] ? 0.85 * d / 2 : 0.0)), f = p * NF;
        this.PF[f + F_C] = c; this.PF[f + F_A] = A;
        this.PF[f + F_D] = Math.min(Math.max(Le / c * fs, 1.0), DL - 4);
        this.PF[f + F_GAIN] = Math.exp(-P.wall * L / d - pack * (P.pack_r || 0));
        this.PF[f + F_LP] = Math.min(0.9, P.hf * L / d + pack * P.pack);
        if (P.wall_kf) {
          // v16.1 generic (set key wall_kf): Kirchhoff visco-thermal wall loss (Kirchhoff 1868; Munjal 2014 eq. 1.x)
          // alpha = (2/d) sqrt(pi f nu) / c (1 + (gamma-1)/sqrt(Pr)), Sutherland viscosity, replaces the flat wall/hf
          // terms: gain + one-pole fitted exactly at 500 Hz and 2 kHz (alpha ~ sqrt f). Packing terms unchanged.
          var Tk = c * c / (G_EX * R_GAS), nu = 1.458e-6 * Tk * Math.sqrt(Tk) / (Tk + 110.4) * R_GAS * Tk / P_ATM;
          var a1 = 2.0 / d * Math.sqrt(Math.PI * 500.0 * nu) / c * (1.0 + (G_EX - 1.0) / Math.sqrt(0.7)) * L * P.wall_kf, a2 = 2.0 * a1;
          var rr = Math.exp(-2.0 * (a2 - a1)), c1 = Math.cos(2 * Math.PI * 500.0 / fs), c2 = Math.cos(2 * Math.PI * 2000.0 / fs);
          var bq = c1 - rr * c2, alk = (bq - Math.sqrt(Math.max(bq * bq - (1 - rr) * (1 - rr), 0.0))) / (1 - rr);
          var lp1 = (1 - alk) / Math.sqrt(1 - 2 * alk * c1 + alk * alk);
          this.PF[f + F_GAIN] = Math.exp(-a1 - pack * (P.pack_r || 0)) / lp1;
          this.PF[f + F_LP] = Math.min(0.9, alk + pack * P.pack);
        }
      }
      if (this.K7 && this.Q) {
        var rpmK = S[S_W] * 60 / (2 * Math.PI), prK = S[S_PMAN] / P_ATM, kk;
        if (!(rpmK > 0)) rpmK = 0; if (!(prK > 0.05)) prK = 0.05;
        for (kk = 0; kk < this.K7.length; kk++) {
          var e7 = this.K7[kk], cq = this.PF[e7[2] * NF + F_C], Tq = cq * cq / (G_EX * R_GAS);
          var q7 = 0.5 * P.eta_v * prK * this.Q[Q_VD] * rpmK / 120 * Tq / T_AMB;
          this.NA[e7[0] * 2 + 1] = e7[1] * q7 / (this.PG[e7[2]][2] * cq) * P.port_bc;
        }
      }
      for (nd = 0; nd < this.NN; nd++) if (this.NODE[nd * this.NW] === 2) {
        var pq = this.NODE[nd * this.NW + 3], a = this.PG[pq][1] / 2, fc = this.PF[pq * NF + F_C] / (2 * Math.PI * a);
        this.NA[nd * 2 + 1] = 1.0 - Math.exp(-2 * Math.PI * Math.min(fc, 0.45 * fs) / fs);
      }
    };
    // v19 generic (set key out_path): per-row outside propagation filter, fitted once per listener as one biquad.
    // Pipe-outlet directivity (Levine & Schwinger 1948 / Ando 1968), ISO 9613-1 air absorption, Delany-Bazley ground
    // reflection (plane wave), and a coherence factor for the ground image from the uncertainty of the microphone
    // geometry (Gaussian phase spread; Salomons 2001, Nord2000 / Plovsing 2006). Row gains keep 1/r and the delay.
    V8pSynth.prototype._outPath = function (rows, pr, srcs) {
      var op = this.P.out_path, fs = this.fs, N = rows.length, F = [], q, j, i;
      for (q = 0; q < 36; q++) F.push(40 * Math.pow(Math.min(16000, 0.45 * fs) / 40, q / 35));
      var RQ = new Float64Array(N * 5), a = (this.P.tail_d || this.c.pipe_d) / 2, ax = op.axis || [-1, 0, 0];
      var Tc = op.T !== undefined ? op.T : 20, RH = op.RH !== undefined ? op.RH : 70;
      this.outFit = [];
      for (j = 0; j < N; j++) {
        var rw = rows[j], s = rw[0], img = rw[5] | 0, ear = pr.ears[rw[1]], pos = srcs[s], z = img ? -pos[2] : pos[2];
        var dx = ear[0] - pos[0], dy = ear[1] - pos[1], dz = ear[2] - z, r = Math.sqrt(dx * dx + dy * dy + dz * dz);
        var th = Math.acos(Math.max(-1, Math.min(1, (dx * ax[0] + dy * ax[1] + dz * (img ? -ax[2] : ax[2])) / r)));
        var useDir = op.dir && s <= 1, sigD = 0.0, sinPsi = (pos[2] + ear[2]) / r;
        if (img && op.ground) this.PATH[j * 5 + 2] /= (pr.ground || 1);
        if (img && op.coh_h) {
          var hs = pos[2], hr = ear[2], dh = Math.sqrt(dx * dx + dy * dy);
          var dl = function (h, d) { return Math.sqrt(d * d + (hs + h) * (hs + h)) - Math.sqrt(d * d + (hs - h) * (hs - h)); };
          var gh = (dl(hr + 0.01, dh) - dl(hr - 0.01, dh)) / 0.02, gd = (dl(hr, dh + 0.01) - dl(hr, dh - 0.01)) / 0.02;
          sigD = Math.sqrt(gh * gh * op.coh_h * op.coh_h + gd * gd * (op.coh_d || 0) * (op.coh_d || 0));
        }
        if (!useDir && !op.air && !img) { RQ[j * 5] = 1; continue; }
        var T = [], imgRow = -1;
        if (!img && op.coh_h) for (i = 0; i < N; i++) if (rows[i][0] === s && rows[i][1] === rw[1] && (rows[i][5] | 0)) imgRow = i;
        for (q = 0; q < F.length; q++) {
          var f = F[q], re = 1.0, im = 0.0;
          if (useDir) re *= lsDir(th, 2 * Math.PI * f * a / C_AIR);
          if (op.air) re *= Math.pow(10, -isoAlpha(f, Tc, RH) * r / 20);
          if (img && op.ground) { var g = dbRefl(f, op.ground, sinPsi), t = re; re = t * g[0]; im = t * g[1]; }
          if (img && sigD > 0) { var k = 2 * Math.PI * f * sigD / C_AIR, cf = Math.exp(-0.5 * k * k); re *= cf; im *= cf; }
          if (imgRow >= 0) {   // incoherent part of the ground image, (1 - C^2) |Q|^2 (r / ri)^2, added in power (Nord2000 energy split)
            var ri_ = rows[imgRow][6], pI = srcs[s], eI = ear, zI = -pI[2], dI = Math.sqrt((eI[0] - pI[0]) * (eI[0] - pI[0]) + (eI[1] - pI[1]) * (eI[1] - pI[1]) + (eI[2] - zI) * (eI[2] - zI));
            var sinI = (pI[2] + eI[2]) / dI, gI = op.ground ? dbRefl(f, op.ground, sinI) : [pr.ground || 0.8, 0], dhI = Math.sqrt((eI[0] - pI[0]) * (eI[0] - pI[0]) + (eI[1] - pI[1]) * (eI[1] - pI[1]));
            var hsI = pI[2], hrI = eI[2], dlI = function (h, d) { return Math.sqrt(d * d + (hsI + h) * (hsI + h)) - Math.sqrt(d * d + (hsI - h) * (hsI - h)); };
            var ghI = (dlI(hrI + 0.01, dhI) - dlI(hrI - 0.01, dhI)) / 0.02, gdI = (dlI(hrI, dhI + 0.01) - dlI(hrI, dhI - 0.01)) / 0.02;
            var sgI = Math.sqrt(ghI * ghI * op.coh_h * op.coh_h + gdI * gdI * (op.coh_d || 0) * (op.coh_d || 0)), kI = 2 * Math.PI * f * sgI / C_AIR, CI = Math.exp(-0.5 * kI * kI);
            var thI = Math.acos(Math.max(-1, Math.min(1, ((eI[0] - pI[0]) * ax[0] + (eI[1] - pI[1]) * ax[1] + (eI[2] - zI) * -ax[2]) / dI)));
            var DI = useDir ? lsDir(thI, 2 * Math.PI * f * a / C_AIR) : 1.0, AI = op.air ? Math.pow(10, -isoAlpha(f, Tc, RH) * dI / 20) : 1.0;
            var eInc = (1 - CI * CI) * (gI[0] * gI[0] + gI[1] * gI[1]) * DI * DI * AI * AI * (r / ri_) * (r / ri_);
            re = Math.sqrt(re * re + eInc);
          }
          var mg = Math.sqrt(re * re + im * im); if (!(mg >= 1e-3)) { if (mg > 1e-30) { re = re / mg * 1e-3; im = im / mg * 1e-3; } else { re = 1e-3; im = 0.0; } }
          T.push([re, im]);
        }
        var Tdb = T.map(function (t) { return 20 * Math.log10(Math.sqrt(t[0] * t[0] + t[1] * t[1]) + 1e-30); });
        var mT = 0, nT = 0, dT = 0; for (q = 0; q < F.length; q++) if (F[q] <= 12000) { mT += Tdb[q]; nT++; } mT /= nT;
        for (q = 0; q < F.length; q++) if (F[q] <= 12000) dT = Math.max(dT, Math.abs(Tdb[q] - mT));
        if (dT < 1.0) { RQ[j * 5] = Math.pow(10, mT / 20); this.outFit.push({ row: j, s: s, img: img, flat: true, r: r }); continue; }   // within 1 dB of flat (e.g. air absorption only, a few metres): gain only
        var b = fitMag(Tdb, F, fs, img ? 'lp' : 'hs'); for (i = 0; i < 5; i++) RQ[j * 5 + i] = b[i];
        // the targets are zero-phase (Gaussian coherence, near-real Rp, real directivity): remove the biquad's low-frequency group delay
        // from the row delay so direct/image interference below the cut-off keeps its geometric phase
        var gB = b[0] + b[1] + b[2], gA = 1 + b[3] + b[4], tau0 = (b[1] + 2 * b[2]) / gB - (b[3] + 2 * b[4]) / gA;
        if (tau0 > 0) this.PATH[j * 5 + 3] = Math.max(1, this.PATH[j * 5 + 3] - tau0);
        this.outFit.push({ row: j, s: s, img: img, th: th, r: r, sigD: sigD, b: b, T: T, F: F });
      }
      this.RQ = RQ; this.RS = new Float64Array(N * 2);
    };
    // v19 generic (set key noise): tyre/road rolling noise (CNOSSOS-EU category 1, octave bands, speed law, tyre width),
    // tyre-cavity resonance c/(pi d_mean) (Sakata et al. 1990), interior wind noise ~ v^n (George 1990, SAE 900315).
    // Pressure in Pa at the ear times the listener gain PV[5], so its level is physical relative to the engine.
    var NOISE_OFF = (typeof process !== 'undefined' && process.env && process.env.NON_ENGINE_NOISE === '0');
    // Cost: one white-noise source through a 3-biquad cascade (low shelf, peak, high shelf) fitted at setup to the target spectrum
    // at 8 table speeds (log-spaced), plus a tyre-cavity peak inside; the level is calibrated per block from the exact target power.
    function nzPeak(f0, gdb, q, fs) {
      var A = Math.pow(10, gdb / 40), w = 2 * Math.PI * f0 / fs, al = Math.sin(w) / (2 * q), cw = Math.cos(w), a0 = 1 + al / A;
      return [(1 + al * A) / a0, -2 * cw / a0, (1 - al * A) / a0, -2 * cw / a0, (1 - al / A) / a0];
    }
    function nzLowShelf(f0, gdb, fs) {
      var A = Math.pow(10, gdb / 40), w = 2 * Math.PI * f0 / fs, cw = Math.cos(w), al = Math.sin(w) / 2 * Math.sqrt(2), sa = 2 * Math.sqrt(A) * al;
      var b0 = A * ((A + 1) - (A - 1) * cw + sa), b1 = 2 * A * ((A - 1) - (A + 1) * cw), b2 = A * ((A + 1) - (A - 1) * cw - sa);
      var a0 = (A + 1) + (A - 1) * cw + sa, a1 = -2 * ((A - 1) + (A + 1) * cw), a2 = (A + 1) + (A - 1) * cw - sa;
      return [b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0];
    }
    V8pSynth.prototype._noiseTarget = function (kmh, F) {   // target pressure PSD (Pa^2/Hz, unweighted) at the ear, per F point
      var o = this.P.noise, R = this.RN, ve = Math.max(kmh, 20), dW = (o.width_db_10mm !== undefined ? o.width_db_10mm : 0.3) * ((o.tyre_w || 195) - 195) / 10;
      var rc = RHO_AIR * C_AIR, D = [], i, q, out = [];
      for (i = 0; i < 8; i++) {   // octave band power -> mean PSD over the octave (bandwidth f / sqrt(2))
        var Lw = CN_AR[i] + CN_BR[i] * Math.log10(ve / 70) + dW - R.TL[i];
        D.push(Math.log10(rc * Math.pow(10, Lw / 10) * 1e-12 / (2 * Math.PI) * R.geo / (CN_F[i] / Math.SQRT2)));
      }
      var Pw = 0.0;
      if (!R.out) Pw = Math.pow(10, ((o.wind_db100 !== undefined ? o.wind_db100 : 65) + 10 * (o.wind_n || 5.5) * Math.log10(Math.max(kmh, 1) / 100) + 10 * Math.log10(o.shape || 1)) / 10) * 4e-10;
      for (q = 0; q < F.length; q++) {
        var x = Math.log2(F[q] / 63), k = Math.max(0, Math.min(6, Math.floor(x))), t = Math.max(-1, Math.min(2, x - k));
        var d = Math.pow(10, D[k] + (D[k + 1] - D[k]) * t);
        out.push(d + Pw * R.windShape[q]);
      }
      return out;
    };
    V8pSynth.prototype._noiseSetup = function (pr) {
      var o = this.P.noise, fs = this.fs, R = { st: new Float64Array(8), dl: new Float32Array(1024), di: 0, seed: 12345, v: 0, k: -1, gain: 0 }, i, q;
      this.RN = R;
      var wb = o.wheelbase || 2.7, xr = o.axle_x !== undefined ? o.axle_x : 1.3, axles = [[xr, 0, 0.05], [xr + wb, 0, 0.05]];
      R.out = !!pr.outside; R.geo = 0.0;
      if (R.out) for (var a = 0; a < 2; a++) { var e = pr.ears[0], dx = e[0] - axles[a][0], dy = e[1] - axles[a][1], dz = e[2] - axles[a][2]; R.geo += 0.5 / Math.max(0.25, dx * dx + dy * dy + dz * dz); }
      else R.geo = 2 * (1 / (o.r_near || 1.0) / (o.r_near || 1.0));   // interior: two tyres per axle side at r_near, then the body TL
      R.TL = CN_F.map(function (f) { return R.out ? 0.0 : Math.max(0.0, 20 * Math.log10((o.floor_m || 9) * f) - 47.0); });
      R.fcav = C_AIR / (Math.PI * (o.d_mean || 0.55));
      R.cav = R.out ? null : nzPeak(R.fcav, o.cav_db !== undefined ? o.cav_db : 6, o.cav_q || 8, fs);
      var F = []; for (q = 0; q < 28; q++) F.push(44 * Math.pow(11300 / 44, q / 27));
      var wbp = rbj('bp', 700.0, 0.5, fs), wlp = rbj('lp', 3000.0, 0.7, fs), wI = 0.0, wS = [];
      for (q = 0; q < F.length; q++) wS.push(Math.pow(10, (bqDb(wbp, F[q], fs) + bqDb(wlp, F[q], fs)) / 10));
      for (q = 1; q < F.length; q++) wI += 0.5 * (wS[q] + wS[q - 1]) * (F[q] - F[q - 1]);
      R.windShape = wS.map(function (w) { return w / wI; }); R.F = F;
      R.tab = [15, 22, 32, 45, 63, 90, 125, 180]; R.bqs = [];
      var self = this;
      R.tab.forEach(function (kmh) {
        var T = self._noiseTarget(kmh, F).map(function (d) { return 10 * Math.log10(d); });
        var p = [150, 0, 1000, 0, 0.7, 4000, 0], lim = [[60, 600], [-30, 30], [200, 6000], [-30, 30], [0.3, 2.5], [1500, 9000], [-30, 30]];
        var mk = function (p) { return [nzLowShelf(p[0], p[1], fs), nzPeak(p[2], p[3], p[4], fs), nzPeak(p[5], p[6], 0.5, fs)]; };
        // only points within 30 dB of the A-weighted maximum count (IEC 61672 A-weighting): the fit follows what is audible
        var TA = T.map(function (t, q) { var f2 = F[q] * F[q], ra = 12194 * 12194 * f2 * f2 / ((f2 + 20.6 * 20.6) * Math.sqrt((f2 + 107.7 * 107.7) * (f2 + 737.9 * 737.9)) * (f2 + 12194 * 12194)); return t + 20 * Math.log10(ra) + 2.0; });
        var mxA = Math.max.apply(null, TA), W = TA.map(function (t) { return t >= mxA - 30 ? 1 : 0; }), nW = W.reduce(function (a, b) { return a + b; }, 0);
        var err = function (p) { var B = mk(p), m = [], off = 0, e = 0;
          for (var q = 0; q < F.length; q++) { var v = bqDb(B[0], F[q], fs) + bqDb(B[1], F[q], fs) + bqDb(B[2], F[q], fs); m.push(v); off += W[q] * (T[q] - v); }
          off /= nW; for (q = 0; q < F.length; q++) { var d = m[q] + off - T[q]; e += W[q] ? d * d : (d > 0 ? d * d : 0); } return e; };   // outside the band: no overshoot
        var be = err(p), step = [0.3, 6, 0.4, 6, 0.4, 0.3, 6];
        for (var it = 0; it < 60; it++) {   // coordinate descent (log steps for frequencies / Q)
          var moved = false;
          for (var j = 0; j < 7; j++) for (var sg = -1; sg <= 1; sg += 2) {
            var t = p.slice(), lg = (j === 0 || j === 2 || j === 4 || j === 5);
            t[j] = lg ? p[j] * Math.exp(sg * step[j]) : p[j] + sg * step[j];
            t[j] = Math.max(lim[j][0], Math.min(lim[j][1], t[j]));
            var et = err(t); if (et < be) { be = et; p = t; moved = true; }
          }
          if (!moved) for (j = 0; j < 7; j++) step[j] *= 0.5;
        }
        var B = mk(p), H2 = 0.0, Lp = [];   // integral of |H|^2 over the fit band (broadband cascade; the cavity peak adds its own power on top)
        for (q = 0; q < F.length; q++) Lp.push(Math.pow(10, (bqDb(B[0], F[q], fs) + bqDb(B[1], F[q], fs) + bqDb(B[2], F[q], fs)) / 10));
        for (q = 1; q < F.length; q++) H2 += 0.5 * (Lp[q] + Lp[q - 1]) * (F[q] - F[q - 1]);
        R.bqs.push({ B: B, H2: H2, rms: Math.sqrt(be / nW) });
      });
      return R;
    };
    V8pSynth.prototype._noiseLevels = function (vms) {   // per block: table shape for this speed and the exact target power
      var R = this.RN, kmh = vms * 3.6, fade = kmh >= 20 ? 1 : (kmh / 20) * (kmh / 20), k = 0, q, F = R.F;
      for (q = 1; q < R.tab.length; q++) if (Math.abs(Math.log(kmh / R.tab[q])) < Math.abs(Math.log(kmh / R.tab[k]))) k = q;
      if (k !== R.k) { R.k = k; R.B = R.bqs[k].B; }
      var T = this._noiseTarget(kmh, F), P = 0.0;
      for (q = 1; q < F.length; q++) P += 0.5 * (T[q] + T[q - 1]) * (F[q] - F[q - 1]);
      R.P = P * fade;
      R.gain = Math.sqrt(R.P / ((1 / 3) * (2 / this.fs) * R.bqs[k].H2));   // uniform noise variance 1/3, one-sided PSD 2 var / fs
    };
    V8pSynth.prototype._noiseRun = function (n, L, R2, off) {
      var R = this.RN, v = this.game.speed || 0;
      R.v += (v - R.v) * Math.min(1, n / (0.25 * this.fs));
      if (R.v < 0.3) return;
      if (!(Math.abs(R.v - (R.vl || 0)) <= 0.005 * R.v)) { this._noiseLevels(R.v); R.vl = R.v; }   // re-level on a 0.5 % speed change (< 0.1 dB)
      var B = R.B, b0 = B[0], b1 = B[1], b2 = B[2], c = R.cav, S = R.st, gain = R.gain * this.PV[5], dl = R.dl, k, seed = R.seed, di = R.di;
      var a0 = b0[0], a1 = b0[1], a2 = b0[2], a3 = b0[3], a4 = b0[4], e0 = b1[0], e1 = b1[1], e2 = b1[2], e3 = b1[3], e4 = b1[4];
      var h0 = b2[0], h1 = b2[1], h2 = b2[2], h3 = b2[3], h4 = b2[4], s0 = S[0], s1 = S[1], s2 = S[2], s3 = S[3], s4 = S[4], s5 = S[5], s6 = S[6], s7 = S[7];
      var c0 = c ? c[0] : 1, c1 = c ? c[1] : 0, c2 = c ? c[2] : 0, c3 = c ? c[3] : 0, c4 = c ? c[4] : 0, g = gain * 2 / 4294967296;
      for (k = 0; k < n; k++) {
        seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; var x = seed - 2147483648;
        var y = a0 * x + s0; s0 = a1 * x - a3 * y + s1; s1 = a2 * x - a4 * y; x = y;
        y = e0 * x + s2; s2 = e1 * x - e3 * y + s3; s3 = e2 * x - e4 * y; x = y;
        y = h0 * x + s4; s4 = h1 * x - h3 * y + s5; s5 = h2 * x - h4 * y; x = y;
        y = c0 * x + s6; s6 = c1 * x - c3 * y + s7; s7 = c2 * x - c4 * y;
        y *= g;
        L[off + k] += y; R2[off + k] += dl[(di - 331) & 1023]; dl[di] = y; di = (di + 1) & 1023;
      }
      S[0] = s0; S[1] = s1; S[2] = s2; S[3] = s3; S[4] = s4; S[5] = s5; S[6] = s6; S[7] = s7; R.seed = seed; R.di = di;
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
          if (s === 5) sh = Math.max(sh, 0.7) * (P.k_st_out !== undefined ? P.k_st_out : 1.0);   // starter: bellhousing / body diffraction, still heard from behind
          rows.push([s, e, sh / Math.max(r, 0.25), r / C_AIR * fs, 0, 0, r]);
          if (s !== 5) {   // starter is under the bellhousing: a specular ground image puts static comb notches (~0.9/2.7/4.5 kHz) in the grind
            var ri = dist(ear, [pos[0], pos[1], -pos[2]]);
            rows.push([s, e, sh * pr.ground / Math.max(ri, 0.25), ri / C_AIR * fs, 0, 1, ri]);
          }
        } else {
          var cabG = s === 5 ? 0.22 * (P.k_st_cab !== undefined ? P.k_st_cab : 1.0) : 1.0;   // starter in the cabin: clearly audible, not 20 dB over idle
          rows.push([s, e, cabG / Math.max(r, 0.25), r / C_AIR * fs, front ? (s === 2 ? 3 : 2) : 1]);
          if (pr.window && (P.cab_model || 0) >= 1) {   // open window: outside free field through the opening
            var sw = front ? P.k_win * (c.eng_shadow !== undefined ? c.eng_shadow : 0.3) : P.k_win;
            rows.push([s, e, cabG * sw / Math.max(r, 0.25), r / C_AIR * fs, 0]);
            var rw = dist(ear, [pos[0], pos[1], -pos[2]]);
            rows.push([s, e, cabG * 0.8 * sw / Math.max(rw, 0.25), rw / C_AIR * fs, 0]);
          }
        }
      }
      this.NPATH = rows.length; this.PATH = new Float64Array(rows.length * 5);
      rows.forEach(function (rw, j) { for (var t = 0; t < 5; t++) this.PATH[j * 5 + t] = rw[t]; }, this);
      this.RQ = null; this.RS = null;
      if (P.out_path && pr.outside) this._outPath(rows, pr, srcs);
      this.RN = (P.noise && !NOISE_OFF) ? this._noiseSetup(pr) : null;
      function coverBq(freq, q0) {
        var q = (freq === 2300.0) ? (P.cover_q1 !== undefined ? P.cover_q1 : q0) : (P.cover_q2 !== undefined ? P.cover_q2 : q0);
        return q <= 0 ? [1, 0, 0, 0, 0] : rbj('bp', freq, q, fs);
      }
      var BQ = [coverBq(2300.0, 10.0), coverBq(4300.0, 14.0), rbj('bp', 280.0, 0.8, fs), rbj('bp', 1800.0, 0.7, fs),
        rbj('lp', P.f_floor, 0.707, fs), rbj('lp', P.f_floor, 0.707, fs), rbj('lp', P.f_fw, 0.707, fs), rbj('lp', P.f_fw, 0.707, fs)];
      if ((P.cab_model || 0) >= 1) { var cr = cabinRows(P, fs); for (var q = 0; q < 4; q++) BQ[4 + q] = cr[q]; }
      var modes = [C_AIR / (2 * c.cabin_len), C_AIR / (2 * c.cabin_h)];
      modes.forEach(function (f) { var b = rbj('bp', f, P.cab_q, fs), h = P.cab_q * 0.5; BQ.push([b[0] * h, b[1] * h, b[2] * h, b[3], b[4]]); });
      var stBq = BQ.length * 5;
      BQ.push(rbj('bp', 850.0, 0.9, fs), rbj('lp', 6500.0, 0.7, fs));   // starter body + open top so mesh / motor / commutator all pass
      this.BQ = new Float64Array(BQ.length * 5);
      BQ.forEach(function (b, j) { for (var t = 0; t < 5; t++) this.BQ[j * 5 + t] = b[t]; }, this);
      if (!keepState || !this.BS) { this.BS = new Float64Array(96); this.H = new Float64Array(NSRC * HB); }
      if (!this.PST) this.PST = new Float64Array(8);
      var fka = C_AIR / (2 * Math.PI * c.pipe_d / 2);
      this.PV = new Float64Array([RHO_AIR / (4 * Math.PI), P.k_int, 0.02 * P.k_tick * (P.tick_scale || 1.0), 5.6 * P.k_st, P.k_cab, 0.25 * P.gain,
        pr.outside ? 0.0 : modes.length, 1.0 - Math.exp(-2 * Math.PI * fka / fs), P.k_hiss,
        1.0 - Math.exp(-2 * Math.PI * 1500.0 / fs), 1.0 - Math.exp(-2 * Math.PI * 7000.0 / fs), P.cab_model || 0, stBq]);
      if (this.Q) {
        var kam = P.k_st_am !== undefined ? P.k_st_am : 0.65;
        if (pr.outside) {
          if (name === 'rear_far' && P.k_st_am_far !== undefined) kam = P.k_st_am_far;
          else if (P.k_st_am_out !== undefined) kam = P.k_st_am_out;
        } else if (P.k_st_am_cab !== undefined) kam = P.k_st_am_cab;
        this.Q[Q_KAM] = kam;
      }
    };
    V8pSynth.prototype.cloneWithListener = function (lis) {
      var o = Object.create(V8pSynth.prototype); Object.assign(o, this);
      ['S', 'CY', 'PF', 'AB', 'BB', 'AT', 'NA', 'K', 'PST', 'Q'].forEach(function (a) { o[a] = this[a].slice(); }, this);
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
    V8pSynth.prototype.reseat = function (rpm, seed) {
      // Keep the compiled network and TMM filters; re-init runtime state (used by offline scoring).
      var i, fo = this.c.fire_order, off = [], left = {};
      this.Q = this._q();
      this.S = new Float64Array(NS);
      this.CY = new Float64Array(this.NCY * NC);
      this.PF = new Float64Array(this.NP * NF);
      this.AB = new Float64Array(this.NP * DL); this.BB = new Float64Array(this.NP * DL); this.AT = new Float64Array(this.NP * DL);
      this.NA = new Float64Array(this.NN * 2);
      seed = seed === undefined ? 1 : seed;
      this.S[S_RNG] = (Math.imul(0x9E3779B9 | 0, seed + 1) >>> 0) || 1;
      this.S[S_PMAN] = P_ATM; this.S[S_TPRIM] = T_AMB; this.S[S_TTAIL] = T_AMB;
      this.c.left_bank.forEach(function (x) { left[x] = 1; });
      for (i = 0; i < this.NCY; i++) off[fo[i] - 1] = 720.0 / this.NCY * i;
      for (i = 0; i < this.NCY; i++) { this.CY[i * NC + C_OFF] = off[i]; this.CY[i * NC + C_BANK] = left[i + 1] ? 0.0 : 1.0; }
      for (var nd = 0; nd < this.NN; nd++) if (this.NODE[nd * this.NW] === 1) this.CY[this.NODE[nd * this.NW + 1] * NC + C_PIPE] = this.NODE[nd * this.NW + 3];
      for (i = 0; i < this.NCY; i++) { var v0 = (0.0 - off[i]) % 720.0; this.CY[i * NC + C_PHI] = v0 < 0 ? v0 + 720.0 : v0; }
      this.pw = 0.0; this.load = 0.0;
      this.K = new Float64Array(NK);
      this.out = new Float64Array(256 * NO);
      this.ctlr = new Ctl(this, false);
      this._intSettled = false; this._settleLeft = 0;
      if (this.tmmCans && this.tmmCans.length) {
        if (this.tmmDual) this.tmmState = this._tmmStates();
        else {
          this.tmmState = [];
          for (i = 0; i < this.tmmCans.length; i++) this.tmmState.push(TMM.makeState(this.tmmCans[i]));
        }
      }
      this.nanResets = 0;
      this._pipes();
      this._listener(this.listener.position || 'rear');
      this.setRunning(rpm || this.c.idle_rpm);
    };
    V8pSynth.prototype._thermal = function (dtb, firing) {
      if (this.P.gas_dyn) { this._thermalGas(dtb, firing); return; }
      var P = this.P, S = this.S, pr = S[S_PMAN] / P_ATM, rpm = S[S_W] * 60 / (2 * Math.PI);
      var tl = firing ? pr * (0.5 + 0.5 * Math.min(1.0, rpm / 4000.0)) : 0.0;
      this.load += (tl - this.load) * Math.min(1.0, dtb / 1.5);
      if (firing) this.pw += (1.0 - this.pw) * dtb / 25.0; else this.pw -= this.pw * dtb / 200.0;
      var tg = firing ? (T_AMB + (P.t_prim0 + P.t_prima * this.load - T_AMB) * (0.6 + 0.4 * this.pw)) : T_AMB + 40.0 * this.pw;
      S[S_TPRIM] += (tg - S[S_TPRIM]) * Math.min(1.0, dtb / 0.4);
      var tt = T_AMB + (0.25 + 0.2 * this.pw) * (S[S_TPRIM] - T_AMB);
      S[S_TTAIL] += (tt - S[S_TTAIL]) * Math.min(1.0, dtb / 1.0);
    };
    // gas_dyn (generic param): the exhaust GAS temperature follows each engine cycle (Heywood 1988, ch. 6: exhaust gas
    // temperature is set per cycle by the expansion; the slow EGT seen on thermocouples is sensor / wall lag). Port gas and
    // load: one engine cycle (120/rpm s). Downstream (tail) gas: plug-flow transport time = exhaust volume downstream of the
    // ports / exhaust volume flow (eta_v * MAP/P_atm * Vd * rpm/120 * T_prim/T_amb), bounded to [one cycle, 2 s].
    // Wall warm-up (pw, 25 s) is unchanged, so steady-state temperatures are the same as without gas_dyn.
    V8pSynth.prototype._thermalGas = function (dtb, firing) {
      var P = this.P, S = this.S, Q = this.Q, pr = S[S_PMAN] / P_ATM, rpm = S[S_W] * 60 / (2 * Math.PI);
      var cyc = 120.0 / (rpm > 300.0 ? rpm : 300.0), kc = Math.min(1.0, dtb / cyc);
      var tl = firing ? pr * (0.5 + 0.5 * Math.min(1.0, rpm / 4000.0)) : 0.0;
      this.load += (tl - this.load) * kc;
      if (firing) this.pw += (1.0 - this.pw) * dtb / 25.0; else this.pw -= this.pw * dtb / 200.0;
      var tg = firing ? (T_AMB + (P.t_prim0 + P.t_prima * this.load - T_AMB) * (0.6 + 0.4 * this.pw)) : T_AMB + 40.0 * this.pw;
      S[S_TPRIM] += (tg - S[S_TPRIM]) * kc;
      var qex = Q[Q_ETAV] * (pr > 0.05 ? pr : 0.05) * Q[Q_VD] * (rpm / 120.0) * (S[S_TPRIM] / T_AMB);
      var ttr = qex > 1e-6 ? (this._vex || 0) / qex : 2.0;
      if (ttr < cyc) ttr = cyc; if (ttr > 2.0) ttr = 2.0;
      var tt = T_AMB + (0.25 + 0.2 * this.pw) * (S[S_TPRIM] - T_AMB);
      S[S_TTAIL] += (tt - S[S_TTAIL]) * Math.min(1.0, dtb / ttr);
    };
    /** one block (n <= 256) -> L/R at off; game = { running, starter, warm, gear } or null (offline 'run' mode) */
    V8pSynth.prototype._tmmStates = function () {
      var st = [], i, d;
      this._tmmSel = [];
      for (i = 0; i < this.tmmCans.length; i++) {
        d = this.tmmDual && this.tmmDual[i];
        var sel = d ? (this._tmmWant(d) || 0) : 0;
        this._tmmSel.push(sel);
        st.push(TMM.makeState(d ? d.comps[sel] : this.tmmCans[i]));
      }
      return st;
    };
    // live mean-flow Mach number at a can, from the engine model (same formula as tools/can_gas.cjs)
    V8pSynth.prototype._tmmMach = function (d) {
      var S = this.S, Q = this.Q, P = this.P;
      if (!S || !Q) return 0;
      var rpm = S[S_W] * 60 / (2 * Math.PI), pr = S[S_PMAN] / P_ATM, tp = S[S_TPRIM];
      var T = d.first ? T_AMB + 0.65 * (tp - T_AMB) : S[S_TTAIL];
      if (!(T > 200)) T = T_AMB;
      var q = 0.5 * P.eta_v * (pr > 0.05 ? pr : 0.05) * Q[Q_VD] * (rpm > 0 ? rpm : 0) / 120 * T / T_AMB;
      return q / d.area / Math.sqrt(G_EX * R_GAS * T);
    };
    V8pSynth.prototype._tmmWant = function (d, cur) {
      var m = this._tmmMach(d);
      if (cur === 1) return m < d.mMid / 1.2 ? 0 : 1;
      if (cur === 0) return m > d.mMid * 1.2 ? 1 : 0;
      return m > d.mMid ? 1 : 0;
    };
    V8pSynth.prototype._tmmSelect = function () {
      var i, st, w, d;
      for (i = 0; i < this.tmmState.length; i++) {
        d = this.tmmDual[i]; st = this.tmmState[i];
        if (!d || !st) continue;
        if (st.dual) { if (st.done) { this.tmmState[i] = st.cur; this._tmmSel[i] = st.want; } continue; }
        w = this._tmmWant(d, this._tmmSel[i]);
        if (w !== this._tmmSel[i]) this.tmmState[i] = { dual: true, cur: st, nxt: TMM.makeState(d.comps[w]), k: 0, want: w, done: false };
      }
    };
    V8pSynth.prototype.block = function (n, rpm, thr, game, L, R, off) {
      if (this.tmmDual) this._tmmSelect();
      if (this._settleLeft > 0) { this._settleStep(n, L, R, off); return; }
      if (this.vcam) this._vc = applyVariableCam(this.Q, this.c, rpm, thr, this._vc, this.P);
      if (this.P.int_res) this.Q[Q_VMAN] = this.c.v_man;   // two-bank tube models the flap; do not also shrink volume
      var dtb = n / this.fs, K = this.ctlr.step(dtb, rpm, thr, game), out = this.out, i;
      for (i = 0; i < n * NO; i++) out[i] = 0.0;
      if (this.ctlr.click) { this.S[29] += this.ctlr.click; this.ctlr.click = 0.0; }   // S_STENV: solenoid / pinion clunk
      if (this.ctlr.silent) { for (i = 0; i < n; i++) { L[off + i] = 0; R[off + i] = 0; } return; }
      // Silently settle the intake on the first audible block (kills a brief start-up burst from a two-bank tube).
      // Duration is a parameter (int_settle, seconds); default 0.25 s when int_res is on, else 0. Uses the current K.
      var settle = (this.P.int_settle !== undefined && this.P.int_settle !== null) ? this.P.int_settle : (this.P.int_res ? 0.25 : 0);
      if (settle && !this._intSettled) {
        // v1.11.43: the 0.25 s pre-roll used to run in one call (about 19 render quanta of work in a single quantum:
        // an audible dropout at key-on / sound switch on a phone). It now runs one 256-sample chunk per call (same
        // chunks, same K, same order) while the block outputs silence, then this block is rendered as before. The
        // controller is held meanwhile, so the result is the old output started 0.125 s later.
        this._intSettled = true;
        this._settleLeft = Math.round(settle * this.fs);
        this._settleStep(n, L, R, off); return;
      }
      this._blockRun(n, K, L, R, off);
    };
    V8pSynth.prototype.settling = function () { return this._settleLeft > 0; };
    V8pSynth.prototype._settleStep = function (n, L, R, off) {
      var K = this.K, out = this.out, j, m = this._settleLeft < 256 ? this._settleLeft : 256;
      if (!this._stL) { this._stL = new Float64Array(256); this._stR = new Float64Array(256); }
      for (j = 0; j < m * NO; j++) out[j] = 0.0;
      kv8(m, this.fs, K, this.Q, this.S, this.CY, this.PF, this.NP, this.NODE, this.NN, this.NW, this.AB, this.BB, this.AT, this.NA, out);
      this._thermal(m / this.fs, K[K_IGN] > 0.5);
      this._pipes();
      post(m, this.fs, out, this.PV, this.PST, this.H, this.PATH, this.NPATH, this.BQ, this.BS, this._stL, this._stR, 0, this.RQ, this.RS);
      this._settleLeft -= m;
      if (this._settleLeft > 0) { for (j = 0; j < n; j++) { L[off + j] = 0; R[off + j] = 0; } return; }
      // pre-roll done: render the held first audible block now (its K; Q is untouched since)
      for (j = 0; j < n * NO; j++) out[j] = 0.0;
      this._blockRun(n, K, L, R, off);
    };
    V8pSynth.prototype._resetNet = function () {
      var i, nAB = this.AB.length, nPF = this.PF.length, nS = this.S.length;
      for (i = 0; i < nAB; i++) { this.AB[i] = 0.0; this.BB[i] = 0.0; this.AT[i] = 0.0; }
      for (i = 0; i < this.NP; i++) { this.PF[i * NF + F_SA] = 0.0; this.PF[i * NF + F_SB] = 0.0; }
      if (this.tmmCans && this.tmmCans.length) {
        if (this.tmmDual) this.tmmState = this._tmmStates();
        else {
          this.tmmState = [];
          for (i = 0; i < this.tmmCans.length; i++) this.tmmState.push(TMM.makeState(this.tmmCans[i]));
        }
      }
      this.nanResets = (this.nanResets || 0) + 1;
    };

    // Test hook: unit impulse at cylinder-1 exhaust port, valves held shut, record mouth U0.
    V8pSynth.prototype.impulseMouth = function (secs) {
      var fs = this.fs, n = Math.round((secs || 0.05) * fs) | 0, i, c, pipe, wi, K, w, m, j;
      this.Q[Q_EVO] = 10000.0;
      this.Q[Q_EVC] = 10001.0;
      for (c = 0; c < this.NCY; c++) this.CY[c * NC + C_OPEN] = 0.0;
      for (i = 0; i < this.AB.length; i++) { this.AB[i] = 0.0; this.BB[i] = 0.0; this.AT[i] = 0.0; }
      for (i = 0; i < this.NP; i++) { this.PF[i * NF + F_SA] = 0.0; this.PF[i * NF + F_SB] = 0.0; }
      if (this.NA) for (i = 0; i < this.NA.length; i++) this.NA[i] = 0.0;
      if (this.tmmCans && this.tmmCans.length) {
        if (this.tmmDual) this.tmmState = this._tmmStates();
        else {
          this.tmmState = [];
          for (i = 0; i < this.tmmCans.length; i++) this.tmmState.push(TMM.makeState(this.tmmCans[i]));
        }
      }
      pipe = this.CY[C_PIPE] | 0;
      K = this.ctlr.step(1 / fs, 520, 0, {});
      for (i = 0; i < NO; i++) this.out[i] = 0.0;
      TMM_RT = this.tmmState;
      kv8(1, fs, K, this.Q, this.S, this.CY, this.PF, this.NP, this.NODE, this.NN, this.NW, this.AB, this.BB, this.AT, this.NA, this.out);
      wi = ((this.S[S_WI] | 0) - 1) & DLM;
      this.AB[pipe * DL + wi] = 1.0;
      var U = new Float64Array(n);
      w = 0;
      while (w < n) {
        m = n - w < 256 ? n - w : 256;
        K = this.ctlr.step(m / fs, 520, 0, {});
        for (j = 0; j < m * NO; j++) this.out[j] = 0.0;
        for (c = 0; c < this.NCY; c++) this.CY[c * NC + C_OPEN] = 0.0;
        kv8(m, fs, K, this.Q, this.S, this.CY, this.PF, this.NP, this.NODE, this.NN, this.NW, this.AB, this.BB, this.AT, this.NA, this.out);
        for (j = 0; j < m; j++) U[w + j] = this.out[j * NO + O_U0];
        w += m;
      }
      TMM_RT = null;
      return U;
    };
    V8pSynth.prototype._blockRun = function (n, K, L, R, off) {
      var out = this.out, dtb = n / this.fs;
      var nf0 = this.S[S_NFIRE];
      TMM_RT = this.tmmState;
      kv8(n, this.fs, K, this.Q, this.S, this.CY, this.PF, this.NP, this.NODE, this.NN, this.NW, this.AB, this.BB, this.AT, this.NA, out);
      TMM_RT = null;
      var zi, nan = false, nOut = n * NO;
      for (zi = 0; zi < nOut; zi++) if (!isFinite(out[zi])) { nan = true; break; }
      if (!nan) {
        for (zi = 0; zi < this.NP; zi++) {
          if (!isFinite(this.PF[zi * NF + F_SA]) || !isFinite(this.PF[zi * NF + F_SB])) { nan = true; break; }
        }
      }
      if (nan) {
        this._resetNet();
        for (zi = 0; zi < nOut; zi++) out[zi] = 0.0;
      }
      var firing = K[K_IGN] > 0.5 && (this.S[S_NFIRE] > nf0 || this.ctlr.recentFire());
      this.ctlr.noteFire(this.S[S_NFIRE] - nf0);
      this._thermal(dtb, firing); this._pipes();
      post(n, this.fs, out, this.PV, this.PST, this.H, this.PATH, this.NPATH, this.BQ, this.BS, L, R, off, this.RQ, this.RS);
      if (this.RN) this._noiseRun(n, L, R, off);
    };
    // ---- EngineVoice / ListenerSource interface
    V8pSynth.prototype.render = function (n, rpm, thr, L, R, off) {
      off = off || 0;
      var gm = this.selfGated ? this.game : null;
      for (var s = 0; s < n;) { var m = Math.min(256, n - s); this.block(m, rpm, thr, gm, L, R, off + s); s += m; }
    };
    V8pSynth.prototype.ownNoise = function () { return !!this.P.noise; };
    V8pSynth.prototype.ctl = function (m) {
      var g = this.game;
      if (m.running !== undefined) { g.running = !!m.running; g.has = true; }
      if (m.starter !== undefined) { g.starter = !!m.starter; g.has = true; }
      if (m.warm !== undefined) g.warm = !!m.warm;
      if (m.gear !== undefined) g.gear = m.gear;
      if (m.speed !== undefined) g.speed = m.speed;   // m/s (v19 road noise; ignored by sets without noise)
    };
    /** the engine was already running when this source was created (model / car switch) */
    V8pSynth.prototype.initRunning = function (on) { if (on) { this.setRunning(this.c.idle_rpm); this.game.running = true; this.game.has = true; } };
    V8pSynth.prototype.report = function () {
      var r = this.ctlr.report() || {};
      r.nanResets = this.nanResets || 0;
      r.pipes = this.NP;
      r.pipes0 = this.NP0 || this.NP;
      r.netLite = !!this.netLite;
      return r;
    };
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
      this.click = this.syn.c.st_kind === 'ford_pe' ? 1.2 : 0.9;   // solenoid pull-in thunk on Bosch / Delco too
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
        var lead = P.st_lead !== undefined ? P.st_lead : (ford ? 0.03 : 0.04);
        if (ford) { if (tc >= lead && this.pin < 0.5) { this.pin = this.sol = 1.0; this.click += 1.6; } }
        else {
          if (tc >= 0.0 && this.pin < 0.5) { this.pin = 1.0; this.click += 1.2; }   // pinion in, quiet low-band lead-in
          if (tc >= lead && this.sol < 0.5) { this.sol = 1.0; this.click += 0.7; }  // motor contacts: first burst
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
      V8pSynth: V8pSynth, DATA: DATA, perspective: perspective, buildNet: buildNet, TMM: TMM };
  }]);

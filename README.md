# Street Rod web demo, v1.1

A browser demo you drive with the arrow keys or, on an iPhone, with on-screen pedals. A procedurally synthesised V8 plays live through a Web Audio AudioWorklet, driven by a longitudinal vehicle model.

* **Single file:** `dist/streetrod.html`. Double-click it to open (it works from `file://`). Rebuild it with `node tools/build-single.mjs`.
* **Deploy-ready folder:** `deploy/` (built by the same script): `index.html` (= the single file), `engine-worklet.js` (same-origin worklet, preferred on https/iOS), `assets/` (manifest, home-screen icons), `_headers`, `.nojekyll`. Upload the folder as-is to any static https host.
* **Multi-file version, no build step:** `index.html` + `js/` + `css/`. This also works from `file://`.

## Controls
| Key | Action |
|---|---|
| Up | Throttle. Hold it; it ramps to full in about 0.7 s |
| Down | Brake. Hold it; it ramps to full pedal in 0.6 s, and full pedal can lock the wheels |
| Right / Left | GT350: shift up / down (the clutch is automatic and stall-free; a downshift that would over-rev is refused). Cadillac: selector N > D > S > L (the Hydra-Matic shifts itself; S holds max 3rd, L holds max 2nd) |
| S or I | Ignition: start / stop (starter cranking, then fast idle) |
| 1 / 2 | Select the 1967 Shelby GT350 / the 1959 Cadillac Series 62 |
| V | View + listening position: Inside car / Outside (rear) (also the on-screen toggle) |
| U | km/h <> mph (speedometer and odometer) |
| T | Reset the trip meter |
| M | Mute |

The **Start engine** button (or any key press / tap) unlocks browser audio.

### Touch (iPhone, landscape)
Brake pedal on the left, gas on the right: hold to press, and press higher up the pedal for more. The two buttons next to the gas shift up/down (GT350) or move the selector N > D > S > L / back to N (Cadillac). The top bar has Start/Stop engine, GT350 | Cadillac, Inside car | Outside (rear), km/h|mph and Mute. Several fingers work at once (gas + brake + shift). Portrait shows a "rotate your phone" card (you can carry on in portrait). Scrolling, pinch zoom and double-tap zoom are blocked. A one-line **Sound:** status under the top bar shows context state, sample rate, worklet or fallback mode, processed blocks, output level and the last audio error.

## What changed in v1.1
* **Inside / Outside.** On-screen toggle plus V. Both the picture and the sound crossfade (about 0.25 s picture, 0.35 s sound).
  * Outside (rear): chase view of the car (GT350 with Le Mans stripes, Cadillac with fins, working brake lights and exhaust puffs) plus a compact tach/speedo/odometer HUD.
  * Inside: per-car period interiors drawn in canvas (no photos), with the car's own live instruments. **GT350:** black dash, wood-rim three-spoke Shelby wheel, deep-set tach (8000) and speedo in twin pods, Stewart-Warner oil/amps gauges under the dash, Hurst T-handle, roll bar in the mirror. **Cadillac:** wide chrome dash, horizontal ribbon speedometer (the red ribbon moves with speed, mph 0-120 or km/h 0-200), big thin-rim wheel with horn ring, column shifter with the P N D S L R quadrant, long hood, fins in the mirror.
  * Cabin sound is now modelled from separate stems of the synth: the exhaust goes through the body (low-passed at 220/480 Hz, cabin boom peak at 58 Hz +9 dB and 115 Hz), the intake/engine-bay/mechanical stem comes through the firewall (low-passed at 2.6 kHz with 420 Hz/1.1 kHz body resonances), with short box reflections, plus road roar rising with v^2 and wind noise rising with v^1.5.
* **iOS audio.** The AudioContext is created and resumed synchronously inside the tap (plus a 1-sample buffer), a silent looping `<audio playsinline>` and `navigator.audioSession.type = 'playback'` make it play with the silent switch on, it resumes after backgrounding (`visibilitychange` / `pageshow`), uses `ctx.sampleRate` (44.1 or 48 kHz), loads the worklet from the same-origin `engine-worklet.js` on http(s) (falling back to Blob, then data: URL), and falls back to a ScriptProcessor running the same DSP if AudioWorklet is missing or produces no blocks within 2.5 s. On-screen buttons act on `pointerup`, and the gesture handlers make no DOM changes (iOS cancels the tap if the page changes during touchstart).
* **Performance.** The voice costs about 0.14 ms per 128-sample block in V8 (about 5 % of one core); canvas DPR is capped at 2 on touch devices. 60 fps in WebKit iPhone emulation, so no quality reduction was needed.

## Architecture
* `js/engine-dsp.js` is a line-by-line port of `engine-sound` v1 (`engine_synth.py`, tag `v1`). It covers the same parameter derivation, the same xorshift64 RNG, and numpy's `RandomState` (MT19937 + legacy gauss) for cylinder imbalance. The output is **bit-exact to Python within float32**: offline worklet render vs Python, max |diff| 4e-13 to 6e-8, SNR 188 to 286 dB on 7 s scripted idle > WOT > lift-off clips for GT350, Cadillac, GT350 open headers and hot-rod Cadillac. Run `test/node_compare.js` and `test/browser_test.mjs`.
  * The worklet wrapper adds the following on top of the v1 model: about 25 ms control smoothing plus per-block linear ramps, a 150 ms crossfade on car change, a 30 ms engine on/off fade, a starter-motor sound, tyre road noise and skid screech, an in-cabin approximation filter, and a stereo-linked look-ahead **limiter** (-1 dBFS ceiling, 3 ms look-ahead, 180 ms release).
  * Loading: the worklet source is inlined. Chrome rejects `blob:` worklets on `file://` pages (the origin is "null") but accepts `data:` URLs, so the code tries `data:` first and falls back to Blob.
* `js/cars.js` holds the engine sound specs, with the **same fields as `EngineSpec` in cars.py**. To drop in v2 specs, `python tools/export_cars.py /workspace/engine-sound > cars.json` dumps them. If v2 changes the kernel itself, `engine-dsp.js` has to be re-ported. The tests compare against a Python reference, so parity can be re-checked.
* `js/vehicles.js` holds drivetrain, chassis and brake data. `js/physics.js` runs the 1 kHz longitudinal model:
  * Engine: a gross torque curve times a net factor, friction and pumping engine braking, about 60 ms manifold lag, a PI idle control with fast idle after start, and an ignition-cut rev limiter with 150 rpm hysteresis.
  * GT350: a stall-free automatic clutch (bite set by rpm and throttle; lock/slip with 560 Nm capacity; it eases when the tyres light up). A shift takes about 0.55 s (declutch and lift, swap, re-engage).
  * Cadillac: a fluid coupling, T = K*we^2*(1-SR^4), stall about 1,900 rpm and no torque multiplication (true for Hydra-Matic). Throttle-dependent up/down shift schedule, kickdown on a fresh floor, coupling refill dip on each shift. It creeps at idle in D.
  * Tyres: rolling constraint, with a slip curve (peak mu at 12 % slip, falling to sliding mu) once grip is exceeded. Rear-axle load includes weight transfer.
  * Brakes: front/rear bias times pedal times capacity times fade. Each axle locks when demand exceeds mu*N (with weight transfer) and stays locked until the pedal comes back below sliding grip. Temperature rises by F*v/heat capacity per axle and cools faster at speed. Drum fade starts at 200 °C (about -60 % at 450 °C); disc fade starts at 350 °C (milder).
  * Aero drag, rolling resistance, and an odometer and trip meter integrated from v*dt.

## Specs used and sources
| | 1967 Shelby GT350 | 1959 Cadillac Series 62 |
|---|---|---|
| Engine | 289 Hi-Po, 306 hp @ 6000, 329 lb-ft @ 4200 (gross), max 6500 rpm [1] | 390, 325 hp @ 4800, 430 lb-ft @ 3100 (gross), 10.5:1 [3][4] |
| Gearbox | **Ford Toploader** 4-speed close ratio 2.32 / 1.69 / 1.29 / 1.00 [1][2] (the T-10, 2.36/1.78/1.41/1.00, was the *1965* GT350 box) | 4-speed Hydra-Matic with controlled fluid coupling, 3.97 / 2.55 / 1.55 / 1.00 [3] (the fact sheet prints 2nd as "2.2553", which is probably a typo for the commonly quoted 2.55) |
| Final drive | 3.89 (4-speed) [1] | 2.94 standard (3.21 optional, standard with A/C) [3][4] |
| Mass | 2,723 lb (1,235 kg) fastback curb [1] + 75 kg driver | Series 62 4,690 to 5,080 lb by body style [3][5]; 4,800 lb (2,180 kg) + 75 kg used |
| Tyres | Goodyear Speedway 350 **E70-15** [1] (the 7.75-15 was the 1965 fitment) | 8.00 x 15 4-ply tubeless (8.20 x 15 whitewall optional) [3][4] |
| Brakes | Power-assisted Kelsey-Hayes 11.3" front discs, 10" x 2.5" rear drums [1][2] | Power-assisted drums all round (12" assumed) [4][5] |
| Weight distribution | 53 / 47 F/R [1] | 53 / 47 assumed |

Assumed or tuned (not from sources): net/gross factor 0.86 (GT350) and 0.80 (Cadillac), Cd*A 0.45*1.95 and 0.55*2.39 m², tyre mu (bias ply) 0.95/0.75 and 0.80/0.64, inertias, the brake capacity and bias, brake thermal masses and fade curves, the clutch capacity and coupling constant.

[1] Shelby Research Group, 1967 GT350/GT500 specifications: https://www.1967shelbyresearch.com/reference/1967-shelby-gt-specifcations.asp
[2] 1967 Shelby spec sheet (2nd ed., 5 Feb 1967): https://www.1967shelbyresearch.com/library/marketing/1967-Shelby-GT-350-GT-500-Spec-Sheet-(2nd-Edition-1967-02-05).pdf
[3] Over-Drive Magazine, 1959 Cadillac fact sheet: https://over-drive-magazine.com/2023/08/29/1959-cadillac-fact-sheet/
[4] 1959 Cadillac brochure, specifications (back cover): http://www2.cadillac.cz/zajimavosti/brochure1959/15.htm
[5] "Cadillac in the 50's" (data tables): https://hoftware.com/erally/articles/Cadillac%20in%20the%2050's.pdf

## Measured (deterministic model, `node test/physics_test.js`; the same in-page result appears in `test/browser_test.mjs`)
The driver feathers the throttle on wheelspin and the GT350 shifts near 6,350 rpm.

| | GT350 | Cadillac |
|---|---|---|
| 0-60 mph | 6.5 s | 9.9 s |
| 0-100 km/h | **6.8 s** | **10.5 s** |
| 1/4 mile | 14.9 s @ 167 km/h | 17.7 s @ 132 km/h |
| Top speed | **203 km/h** (126 mph, limiter-capped in 4th) | **197 to 198 km/h** (123 mph, near peak power in 4th) |
| 100-0 km/h, threshold braking (cold) | **48.2 m** (3.5 s) | **60.0 m** (4.4 s) |
| 100-0 km/h, full pedal (locks up) | 51.4 m | 60.0 m |
| Fade: 8 threshold stops from 100, 20 s at 100 km/h between | 48.2 m > 50.1 m (front discs stay below 210 °C, rear drums reach 280 °C) | 60.0 m > 80.0 m (drums reach 330 to 370 °C) |

Real-time keyboard runs in headless Chromium: GT350 0-100 in about 8.2 s (a flat-out keyboard launch spins the tyres), Cadillac about 11.0 s.

## Tests
`node test/node_compare.js` (port vs Python), `node test/physics_test.js`, `node test/browser_test.mjs [dist/streetrod.html|index.html]` (headless Chrome: console errors, file:// worklet, offline worklet vs Python, limiter, shifts, odometer, mute, screenshots), and `node test/ios_audio_test.mjs` (WebKit with iPhone 15 landscape / iPhone 14 portrait emulation plus Chromium desktop, over http://localhost and file://: sound starts from a tap, worklet and forced ScriptProcessor modes, taps on view toggle/shift/car, gas and brake pedals, fps, no console errors; writes the iPhone screenshots), and `node test/render_drive.mjs` (full chain offline render: peak stays at or below -1 dBFS; writes `demo-audio/*.mp3`).
Python reference: `python test/ref_render.py ref test/out/traj.json test/out`. `ref/` is a frozen copy of engine-sound tag v1.

## Known limitations
* Longitudinal physics for drag and cruising (the road scene is a decoration that scrolls with real distance; A / D only moves the car across the lanes). The Road course mode (key C) adds steering and corners on authored roads (js/road.js): a lateral grip budget on top of the same longitudinal solver. Since v1.13.34 (prototype) the default is a Lotus Turbo Challenge 2-style stage (Mesa Run: hills, checkpoints with a countdown, three rivals, traffic, objects) with a generic collision engine (js/collide.js, js/road-traffic.js) and a pseudo-3D renderer (js/road3d.js); ?course=mesa keeps the flat v1.13.32 lap.
* The in-cabin sound is a stem-based model (exhaust through the body, intake/mechanical through the firewall, boom, road and wind), not a measured cabin response.
* Tested on WebKit with iPhone 14/15 emulation, not on a real iPhone. The status line is there to diagnose a real device.
* The GT350 clutch is automated (there is no clutch key). A flat-out keyboard launch in 1st lights up the tyres; feather Up for a clean start.
* The open differential (one-wheel peel), suspension squat/dive, gear-whine and drivetrain lash are not modelled. There is no reverse or park.
* Brake fade thresholds, thermal masses and tyre grip are plausible estimates, not measured period data.
* Gross > net power is a single factor. The torque curves are smooth fits through the two published points.
* v1 synth quirk, kept for parity: in `engine_synth.py` the bank-1 crossover reads bank 0's *tailpipe* output, because `ys0` has already been overwritten. The fix belongs in the Python v2 first.

## Hosting publicly (not done)
It is a static site. Upload `deploy/` as-is. iPhone needs **https** (AudioWorklet and audio unlock need a secure context; `http://localhost` is the only exception).
* **GitHub Pages** (needs a free GitHub account): push `deploy/` (or the repo, with Pages set to that folder) and turn on Settings > Pages. Served at `https://<user>.github.io/<repo>/`. Durable; best for sharing.
* **Netlify / Cloudflare Pages** (free account): drag `deploy/` onto the dashboard. No build command.
* **No account, temporary:**
  * **Netlify Drop** (https://app.netlify.com/drop) or `npx netlify-cli deploy --allow-anonymous --dir deploy`: gives an https URL straight away, but the site is password-protected until claimed and is deleted if it isn't claimed (with an account) within about an hour.
  * **Cloudflare quick tunnel** (`cloudflared tunnel --url http://localhost:8000` while `python3 -m http.server` serves `deploy/`): a random `https://*.trycloudflare.com` URL with no account, alive only while the box runs the command. Good for a quick iPhone test.

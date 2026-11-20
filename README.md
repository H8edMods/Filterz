# Filterz 📸
Snapchat-style camera app that runs in **Android Chrome** (installable PWA). 80+ live filters: face AR (spiders, old age, skull paint, devil, cat, dog, bunny, crowns, hats, robot mesh, fire/rainbow breath, tears...), face warps (big eyes, big mouth, tiny face...), color/film/glitch looks, and screen effects (confetti, snow, kaleidoscope, swirl...).
Tap shutter = photo, hold = video (with mic). Files save to Downloads.

## Run
- **GitHub Pages:** push this folder, Settings → Pages → deploy from `main` / root. Open the URL in Chrome on Android → ⋮ → *Install app*.
- **Termux:** `python -m http.server 8080` in this folder, open `http://localhost:8080` in Chrome (camera works on localhost).
Camera needs HTTPS or localhost. Face tracking (MediaPipe Face Landmarker, pinned v0.10.14) loads from CDN on first launch, so internet is required.

## Add filters
Edit the `FL` array in `app.js`. Shader looks: add a mode number to the fragment shader. AR: `ar(icon,name,(face,time)=>{...})` using `st()` emoji stickers or `loc()` vector shapes.

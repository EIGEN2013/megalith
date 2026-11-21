
(function () {
  'use strict';
  var $ = function (id) { return document.getElementById(id); };
  var curtain = $('curtain');
  function fail(message) { curtain.hidden = false; $('curtain-text').textContent = message; }
  if (!window.THREE || !THREE.GLTFLoader) { fail('3D 라이브러리를 불러오지 못했습니다. 네트워크 연결을 확인한 뒤 다시 열어 주세요.'); return; }

  var WORLD = JSON.parse($('world-json').textContent);
  var STORY = JSON.parse($('story-json').textContent);

  // ---------------------------------------------------------------- tuning (mirrors player.gd)
  var WALK_SPEED = 1.7, RUN_SPEED = 5.2, ACCEL = 28, AIR_ACCEL = 10, TURN_SPEED = 12;
  var GRAVITY = 20, JUMP_VELOCITY = 7.0, DOUBLE_JUMP_VELOCITY = 6.6, MAX_JUMPS = 2;
  var WALK_CLIP_SPEED = 1.5, RUN_CLIP_SPEED = 4.4, ATTACK_SPEED = 1.0, BLEND = 0.15;   // match the re-authored clips
  var RADIUS = 0.32, HEIGHT = 1.8, STEP_UP = 0.35;

  // ---------------------------------------------------------------- terrain + colliders
  var HM = WORLD.height, heights = (function () {
    var bin = atob(HM.int16_le_base64), view = new DataView(new ArrayBuffer(bin.length));
    for (var i = 0; i < bin.length; i++) view.setUint8(i, bin.charCodeAt(i));
    var out = new Float32Array(bin.length / 2);
    for (var k = 0; k < out.length; k++) out[k] = view.getInt16(k * 2, true) * HM.unit;
    return out;
  })();
  var zone = 'forest', VILLAGE = null;
  function groundAt(x, z) {
    if (zone === 'village') {
      if (LAND.ready && (Math.abs(x) > 63 || Math.abs(z) > 63 || Math.hypot(x, z - 3) > 31)) return LAND.groundAt(x, z);
      return VILLAGE.groundAt(x, z);
    }
    var fx = Math.max(0, Math.min(HM.size - 1.001, (x + HM.half) / HM.step));
    var fz = Math.max(0, Math.min(HM.size - 1.001, (z + HM.half) / HM.step));
    var ix = Math.floor(fx), iz = Math.floor(fz), tx = fx - ix, tz = fz - iz, row = iz * HM.size + ix;
    var a = heights[row] * (1 - tx) + heights[row + 1] * tx;
    var b = heights[row + HM.size] * (1 - tx) + heights[row + HM.size + 1] * tx;
    return a * (1 - tz) + b * tz;
  }
  function boxesOf(list) {
    return list.map(function (c) { return { x: c.x, z: c.z, cos: Math.cos(c.yaw), sin: Math.sin(c.yaw), hx: c.hx, hz: c.hz, y0: c.y0, y1: c.y1 }; });
  }
  var BOXES = boxesOf(WORLD.colliders);
  function toLocal(b, x, z) { var dx = x - b.x, dz = z - b.z; return [dx * b.cos - dz * b.sin, dx * b.sin + dz * b.cos]; }
  function floorAt(x, z, feetY) {
    var best = groundAt(x, z);
    for (var i = 0; i < BOXES.length; i++) {
      var b = BOXES[i];
      if (b.y1 <= best || b.y1 > feetY + STEP_UP) continue;
      var l = toLocal(b, x, z), inset = RADIUS * 0.5;
      if (Math.abs(l[0]) < b.hx + inset && Math.abs(l[1]) < b.hz + inset) best = b.y1;
    }
    return best;
  }
  function pushOut(p, v) {
    for (var i = 0; i < BOXES.length; i++) {
      var b = BOXES[i];
      if (p.y >= b.y1 - 0.05 || p.y + HEIGHT <= b.y0) continue;
      if (b.y1 - p.y <= STEP_UP) continue;            // a low ledge (a step, the bridge's end): stepped onto, not a wall
      var l = toLocal(b, p.x, p.z), ex = b.hx + RADIUS, ez = b.hz + RADIUS;
      if (Math.abs(l[0]) >= ex || Math.abs(l[1]) >= ez) continue;
      var px = ex - Math.abs(l[0]), pz = ez - Math.abs(l[1]);
      if (px < pz) l[0] = l[0] < 0 ? -ex : ex; else l[1] = l[1] < 0 ? -ez : ez;
      p.x = b.x + l[0] * b.cos + l[1] * b.sin;
      p.z = b.z - l[0] * b.sin + l[1] * b.cos;
      v.x *= 0.6; v.z *= 0.6;
    }
    landPush(p, v);
    if (zone === 'village') {           // villagers are solid too
      for (var k = 0; k < NPCS.length; k++) {
        var n = NPCS[k], dx = p.x - n.pos.x, dz = p.z - n.pos.z, d = Math.hypot(dx, dz), reach = RADIUS + 0.3 * (n.def.scale || 1);
        if (d < reach && d > 1e-4) { p.x = n.pos.x + dx / d * reach; p.z = n.pos.z + dz / d * reach; }
      }
    }
  }

  // ---------------------------------------------------------------- renderer + sky + light
  var canvas = $('view'), stage = $('stage');
  var renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, powerPreference: 'high-performance' });   // the discrete GPU on laptops
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  function srgb(hex) { return new THREE.Color(hex).convertSRGBToLinear(); }   // authored in sRGB, lit in linear

  // ---------------------------------------------------------------- surface detail
  // Every vertex-coloured surface gets a tileable relief texture by material name, projected
  // triplanar (no UVs needed): mortar between ashlar blocks, wood grain, thatch fibres, slate
  // courses, cobbles. The detail colour is normalised to its mean, so the model's own colours stay.
  var TEXMETA = window.MEGALITH_TEX || {};
  var DETAIL = { tex: {}, loader: new THREE.TextureLoader() };
  var DETAIL_BY_MAT = {
    stone: ['fieldstone'], rock: ['rock'], slate: ['slate'], thatch: ['thatch'], straw: ['thatch'], wood: ['wood', 'planks'],
    plaster: ['plaster'], cobble: ['cobble'], ground: ['ground'], grass: ['grass'], iron: ['iron'], cloth: ['cloth'],
    ceramic: ['smooth'], gold: ['iron'], paper: ['smooth'], produce: ['smooth'], food: ['smooth']
  };
  function detailTex(name) {
    if (!DETAIL.tex[name]) {
      var pair = {};
      ['d', 'n'].forEach(function (k) {
        var t = DETAIL.loader.load('tex/' + name + '_' + k + '.jpg');
        t.wrapS = t.wrapT = THREE.RepeatWrapping;
        t.anisotropy = renderer.capabilities.getMaxAnisotropy();
        pair[k] = t;
      });
      DETAIL.tex[name] = pair;
    }
    return DETAIL.tex[name];
  }
  function addDetail(material, side, top, opts) {
    if (!material || material.userData.detail || material.map) return;
    opts = opts || {};
    material.userData.detail = side + '/' + (top || side);
    var A = detailTex(side), T = detailTex(top || side);
    var ma = TEXMETA[side] || { tile: 2, mean: 0.8 }, mt = TEXMETA[top || side] || ma;
    var uni = {
      uTriD: { value: A.d }, uTriN: { value: A.n }, uTriDY: { value: T.d }, uTriNY: { value: T.n },
      uTriS: { value: 1 / ma.tile }, uTriSY: { value: 1 / mt.tile }, uTriMean: { value: Math.pow(ma.mean, 1.0) }, uTriMeanY: { value: mt.mean },
      uTriStr: { value: opts.strength || 1.0 }, uTriAmt: { value: opts.amount == null ? 1.0 : opts.amount }
    };
    var prev = material.onBeforeCompile;
    material.onBeforeCompile = function (sh, r) {
      if (prev) prev.call(this, sh, r);
      Object.keys(uni).forEach(function (k) { sh.uniforms[k] = uni[k]; });
      // world space: the village meshes are quantised (their scale lives on the node), so object space is useless
      sh.vertexShader = 'varying vec3 vTriP; varying vec3 vTriN;\n' +
        sh.vertexShader.replace('#include <project_vertex>', [
          '#include <project_vertex>',
          'vec4 triWP = vec4(transformed, 1.0);',
          'mat3 triM = mat3(modelMatrix);',
          '#ifdef USE_INSTANCING', 'triWP = instanceMatrix * triWP; triM = triM * mat3(instanceMatrix);', '#endif',
          'vTriP = (modelMatrix * triWP).xyz;',
          'vTriN = normalize(triM * normal);'].join('\n'));
      sh.fragmentShader = [
        'uniform sampler2D uTriD; uniform sampler2D uTriN; uniform sampler2D uTriDY; uniform sampler2D uTriNY;',
        'uniform float uTriS; uniform float uTriSY; uniform float uTriMean; uniform float uTriMeanY; uniform float uTriStr; uniform float uTriAmt;',
        'varying vec3 vTriP; varying vec3 vTriN;'].join('\n') + '\n' +
        sh.fragmentShader
          .replace('#include <color_fragment>', [
            '#include <color_fragment>',
            'vec3 tNo = normalize(vTriN);',
            'vec3 tW = pow(abs(tNo), vec3(4.0)); tW /= (tW.x + tW.y + tW.z);',
            'vec3 tP = vTriP;',
            'vec3 tAlb = pow(texture2D(uTriD, tP.zy * uTriS).rgb, vec3(2.2)) * tW.x / uTriMean',
            '          + pow(texture2D(uTriDY, tP.xz * uTriSY).rgb, vec3(2.2)) * tW.y / uTriMeanY',
            '          + pow(texture2D(uTriD, tP.xy * uTriS).rgb, vec3(2.2)) * tW.z / uTriMean;',
            'diffuseColor.rgb *= mix(vec3(1.0), tAlb, uTriAmt);',
            // the old village ground dissolves into the scanned land over a wide, irregular band, in fine grain
            material.userData.clipR ? ['{ vec2 cq = vTriP.xz - vec2(0.0, 3.0);',
              '  float grain = fract(sin(dot(floor(vTriP.xz * 12.0), vec2(12.9898, 78.233))) * 43758.5453);',
              '  float swell = 0.5 + 0.25 * sin(vTriP.x * 0.21 + sin(vTriP.z * 0.17) * 2.0) + 0.25 * sin(vTriP.z * 0.33 - vTriP.x * 0.12);',
              '  if (length(cq) > ' + material.userData.clipR.toFixed(1) + ' - 6.0 * (0.55 * swell + 0.45 * grain)) discard; }'].join('\n') : ''].join('\n'))
          .replace('#include <normal_fragment_maps>', [
            '#include <normal_fragment_maps>',
            '{',
            '  vec3 nX = texture2D(uTriN, tP.zy * uTriS).xyz * 2.0 - 1.0;',
            '  vec3 nY = texture2D(uTriNY, tP.xz * uTriSY).xyz * 2.0 - 1.0;',
            '  vec3 nZ = texture2D(uTriN, tP.xy * uTriS).xyz * 2.0 - 1.0;',
            '  nX.xy *= uTriStr; nY.xy *= uTriStr; nZ.xy *= uTriStr;',
            '  nX = vec3(nX.xy + tNo.zy, abs(nX.z) * tNo.x);',
            '  nY = vec3(nY.xy + tNo.xz, abs(nY.z) * tNo.y);',
            '  nZ = vec3(nZ.xy + tNo.xy, abs(nZ.z) * tNo.z);',
            '  vec3 objN = normalize(nX.zyx * tW.x + nY.xzy * tW.y + nZ.xyz * tW.z);',
            '  vec3 dn = normalize((viewMatrix * vec4(objN, 0.0)).xyz);',
            '  #ifdef DOUBLE_SIDED', '  dn *= (gl_FrontFacing ? 1.0 : -1.0);', '  #endif',
            '  normal = dn;',
            '}'].join('\n'));
    };
    var key = material.customProgramCacheKey ? material.customProgramCacheKey.bind(material) : function () { return ''; };
    material.customProgramCacheKey = function () { return key() + '|tri' + (material.userData.clipR || ''); };
    material.needsUpdate = true;
  }
  function detailFor(material, overrides) {
    if (!material || material.map) return;
    var name = (material.name || '').replace(/\.\d+$/, '');
    var pick = (overrides && overrides[name]) || DETAIL_BY_MAT[name];
    if (pick) addDetail(material, pick[0], pick[1], pick[2]);
  }
  function detailTree(root, overrides) {
    root.traverse(function (o) {
      if (!o.isMesh) return;
      var mats = Array.isArray(o.material) ? o.material : [o.material];
      mats.forEach(function (m) { detailFor(m, overrides); });
    });
  }

  var scene = new THREE.Scene();
  scene.background = srgb(0xc4d6dc);
  scene.fog = new THREE.Fog(srgb(0xc4d6dc), 30, 165);
  var camera = new THREE.PerspectiveCamera(55, 1, 0.1, 900);

  var sky = new THREE.Mesh(
    new THREE.SphereGeometry(600, 24, 12),
    new THREE.ShaderMaterial({
      side: THREE.BackSide, depthWrite: false, fog: false,
      uniforms: { top: { value: new THREE.Color(0x3f74b8) }, low: { value: new THREE.Color(0xc4d6dc) } },
      vertexShader: 'varying float h; void main(){ h = normalize(position).y; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
      fragmentShader: 'uniform vec3 top; uniform vec3 low; varying float h; void main(){ gl_FragColor = vec4(mix(low, top, smoothstep(0.02, 0.6, h)), 1.0); }'
    }));
  sky.renderOrder = -1;
  scene.add(sky);

  // The ruins share the village's light: a warm low sun, soft sky fill, wide crisp shadows.
  var hemi = new THREE.HemisphereLight(srgb(0xe2ecff), srgb(0x6b7454), 0.78);
  scene.add(hemi);
  var sun = new THREE.DirectionalLight(srgb(0xffe6c4), 1.4);
  sun.castShadow = true;
  sun.shadow.mapSize.set(renderer.capabilities.maxTextureSize >= 4096 ? 4096 : 2048, renderer.capabilities.maxTextureSize >= 4096 ? 4096 : 2048);
  sun.shadow.camera.left = -22; sun.shadow.camera.right = 22;
  sun.shadow.camera.top = 22; sun.shadow.camera.bottom = -22;
  sun.shadow.camera.near = 1; sun.shadow.camera.far = 80;
  sun.shadow.bias = -0.0012; sun.shadow.normalBias = 0.03;
  scene.add(sun, sun.target);
  var SUN_OFFSET = new THREE.Vector3(10, 22, 14);

  // Filmic tone mapping, and the sky itself as image-based light: every surface picks up soft
  // sky reflections and fill from the right directions, rather than a flat ambient term.
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.15;
  (function skyLight() {
    if (/noenv/.test(location.search)) return;
    var W = 128, H = 64, px = new Uint8Array(W * H * 4);
    var top = srgb(0x5d8fd0), low = srgb(0xd8e2e6), gnd = srgb(0x5e5a46), sd = SUN_OFFSET.clone().normalize();
    for (var j = 0; j < H; j++) {
      var lat = (0.5 - (j + 0.5) / H) * Math.PI, y = Math.sin(lat);
      for (var i = 0; i < W; i++) {
        var lon = ((i + 0.5) / W) * 2 * Math.PI - Math.PI;
        var dx = Math.cos(lat) * Math.sin(lon), dz = Math.cos(lat) * -Math.cos(lon);
        var t, c = new THREE.Color();
        if (y > 0) { t = Math.min(1, y / 0.7); t = t * t * (3 - 2 * t); c.copy(low).lerp(top, t); }
        else { t = Math.min(1, -y / 0.25); c.copy(low).lerp(gnd, t * t * (3 - 2 * t)); }
        var s = Math.pow(Math.max(0, dx * sd.x + y * sd.y + dz * sd.z), 24) * 0.6;
        var k = (j * W + i) * 4;
        px[k] = Math.min(255, (c.r + s) * 255); px[k + 1] = Math.min(255, (c.g + s * 0.85) * 255); px[k + 2] = Math.min(255, (c.b + s * 0.6) * 255); px[k + 3] = 255;
      }
    }
    var eq = new THREE.DataTexture(px, W, H, THREE.RGBAFormat);
    eq.mapping = THREE.EquirectangularReflectionMapping;
    eq.magFilter = eq.minFilter = THREE.LinearFilter;
    eq.flipY = true;
    eq.needsUpdate = true;
    var pm = new THREE.PMREMGenerator(renderer);
    scene.environment = pm.fromEquirectangular(eq).texture;
    pm.dispose();
    eq.dispose();
    hemi.intensity = 0.3;
  })();

  // ---------------------------------------------------------------- input
  var keys = Object.create(null);
  var pressed = { jump: false, attack: false, interact: false };
  var KEYMAP = {
    KeyW: 'forward', ArrowUp: 'forward', KeyS: 'back', ArrowDown: 'back', KeyA: 'left', ArrowLeft: 'left',
    KeyD: 'right', ArrowRight: 'right', ShiftLeft: 'run', ShiftRight: 'run', Space: 'jump', KeyJ: 'attack',
    KeyF: 'interact', Enter: 'confirm', KeyQ: 'camLeft', KeyE: 'camRight'
  };
  window.addEventListener('keydown', function (e) {
    var action = KEYMAP[e.code];
    if (!action) return;
    e.preventDefault();
    if (!e.repeat && (action === 'jump' || action === 'interact' || action === 'confirm')) {
      if (talk.active) { advance(); return; }
      if (action === 'confirm') return;
    }
    if (!e.repeat && pressed.hasOwnProperty(action)) pressed[action] = true;
    keys[action] = true;
  });
  window.addEventListener('keyup', function (e) { var action = KEYMAP[e.code]; if (action) keys[action] = false; });
  window.addEventListener('blur', function () { keys = Object.create(null); });

  var cam = { yaw: 0, pitch: 0.42, distance: 3.4, goal: 3.4, indoor: 0 };
  var drag = null;
  canvas.addEventListener('pointerdown', function (e) {
    canvas.focus();
    drag = { id: e.pointerId, x: e.clientX, y: e.clientY, moved: 0, button: e.button, touch: e.pointerType === 'touch' };
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointermove', function (e) {
    if (!drag || drag.id !== e.pointerId) return;
    var dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    drag.x = e.clientX; drag.y = e.clientY; drag.moved += Math.abs(dx) + Math.abs(dy);
    cam.yaw -= dx * 0.006;
    cam.pitch = Math.max(-0.25, Math.min(1.25, cam.pitch + dy * 0.005));
  });
  function endDrag(e) {
    if (!drag || drag.id !== e.pointerId) return;
    if (e.type === 'pointerup' && drag.moved < 6) {
      if (talk.active) advance();
      else if (drag.button === 0 && !drag.touch) pressed.attack = true;   // a click that was not a drag
    }
    drag = null;
  }
  canvas.addEventListener('pointerup', endDrag);
  canvas.addEventListener('pointercancel', endDrag);
  canvas.addEventListener('contextmenu', function (e) { e.preventDefault(); });
  canvas.addEventListener('wheel', function (e) {
    e.preventDefault();
    cam.goal = Math.max(2.5, Math.min(10, cam.goal + (e.deltaY > 0 ? 0.6 : -0.6)));
  }, { passive: false });

  var stickVec = { x: 0, y: 0 };
  var stick = $('stick'), knob = $('knob');
  function showTouch() { document.body.classList.add('touching'); }
  window.addEventListener('touchstart', showTouch, { passive: true });
  if (window.matchMedia && window.matchMedia('(pointer: coarse)').matches) showTouch();
  function stickMove(e) {
    var r = stick.getBoundingClientRect();
    var x = (e.clientX - (r.left + r.width / 2)) / (r.width / 2), y = (e.clientY - (r.top + r.height / 2)) / (r.height / 2);
    var len = Math.hypot(x, y);
    if (len > 1) { x /= len; y /= len; }
    stickVec.x = x; stickVec.y = y;
    knob.style.transform = 'translate(' + (x * 33) + 'px,' + (y * 33) + 'px)';
  }
  stick.addEventListener('pointerdown', function (e) { stick.setPointerCapture(e.pointerId); stickMove(e); });
  stick.addEventListener('pointermove', function (e) { if (stick.hasPointerCapture(e.pointerId)) stickMove(e); });
  function stickEnd() { stickVec.x = 0; stickVec.y = 0; knob.style.transform = ''; }
  stick.addEventListener('pointerup', stickEnd);
  stick.addEventListener('pointercancel', stickEnd);
  $('btn-jump').addEventListener('pointerdown', function (e) { e.preventDefault(); pressed.jump = true; });
  $('btn-attack').addEventListener('pointerdown', function (e) { e.preventDefault(); pressed.attack = true; });
  function onActivate(el, fn) {
    el.addEventListener('click', fn);
    el.addEventListener('keydown', function (e) { if (e.code === 'Enter' || e.code === 'Space') { e.preventDefault(); e.stopPropagation(); fn(); } });
  }
  onActivate($('prompt'), function () { pressed.interact = true; });
  onActivate($('dialogue'), function () { advance(); });

  // ---------------------------------------------------------------- dialogue
  var talk = { active: false, lines: [], index: 0, shown: 0, done: null };
  var dlg = $('dialogue'), dlgWho = $('dlg-who'), dlgText = $('dlg-text');
  var unlockAt = 0, clockNow = 0;

  function say(lines, done) {
    if (!lines || !lines.length) { if (done) done(); return; }
    talk.active = true; talk.lines = lines; talk.index = 0; talk.done = done || null;
    hero.locked = true;
    stage.classList.add('talking');
    dlg.hidden = false;
    showLine();
  }
  function showLine() {
    var line = talk.lines[talk.index];
    talk.shown = 0;
    dlg.dataset.who = line.who;
    dlg.classList.remove('ready');
    dlgWho.textContent = STORY.speakers[line.who] || '';
    dlgText.textContent = '';
    if (line.stage) onStage(line.stage);
  }
  function advance() {
    if (!talk.active) return;
    var line = talk.lines[talk.index];
    if (talk.shown < line.text.length) { talk.shown = line.text.length; dlgText.textContent = line.text; dlg.classList.add('ready'); return; }
    talk.index += 1;
    if (talk.index < talk.lines.length) { showLine(); return; }
    talk.active = false;
    dlg.hidden = true;
    stage.classList.remove('talking');
    pressed.jump = pressed.attack = pressed.interact = false;
    unlockAt = clockNow + 0.2;          // so the key that closed the box does not also jump
    var done = talk.done; talk.done = null;
    if (done) done();
  }
  function updateTalk(dt) {
    if (!talk.active) return;
    var line = talk.lines[talk.index];
    if (talk.shown >= line.text.length) return;
    talk.shown = Math.min(line.text.length, talk.shown + dt * 42);
    dlgText.textContent = line.text.slice(0, Math.floor(talk.shown));
    if (talk.shown >= line.text.length) dlg.classList.add('ready');
  }

  // ---------------------------------------------------------------- hero
  var hero = { root: null, mixer: null, actions: {}, current: '', pos: new THREE.Vector3(), vel: new THREE.Vector3(),
               heading: WORLD.spawn.heading, onFloor: true, jumpsUsed: 0, attacking: false, flipping: false,
               locked: true, cutscene: true };
  hero.pos.set(WORLD.spawn.x, groundAt(WORLD.spawn.x, WORLD.spawn.z), WORLD.spawn.z);

  function play(name, fade, timeScale) {
    var next = hero.actions[name];
    if (hero.current === name) { if (timeScale) next.timeScale = timeScale; return; }
    var prev = hero.actions[hero.current];
    next.reset();
    next.timeScale = timeScale || 1;
    next.setEffectiveWeight(1);
    next.play();
    if (prev) prev.crossFadeTo(next, fade, false);
    hero.current = name;
  }
  function replay(name, fade, timeScale) { hero.current = hero.current === name ? '' : hero.current; play(name, fade, timeScale); }

  function parseGlb(id, keepBitmaps, done) {
    var bin = atob($(id).textContent.trim()), bytes = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    // The <img> texture path works from blob: URLs everywhere this page is embedded.
    var saved = window.createImageBitmap;
    if (!keepBitmaps) { try { window.createImageBitmap = undefined; } catch (err) { /* read-only */ } }
    function restore() { try { window.createImageBitmap = saved; } catch (err) { /* ignore */ } }
    new THREE.GLTFLoader().parse(bytes.buffer, '', function (gltf) { restore(); done(gltf); },
      function (err) { restore(); fail('모델을 읽지 못했습니다: ' + (err && err.message ? err.message : err)); });
  }

  // ---------------------------------------------------------------- world
  var fx = { runes: [], memory: [], portal: null, pulse: 0, beacons: [] };
  var found = [false, false, false], foundCount = 0, finished = false;
  var flags = { left: false, edgeAt: -99 };
  var POINTS = WORLD.interact;
  var forestObjs = [], forestScene = null, doorNode = null;

  function setupWorld(gltf) {
    var world = gltf.scene;
    world.traverse(function (o) {
      if (!o.isMesh) return;
      var n = o.name, m = o.material;
      if (n.indexOf('terrain') === 0) {
        o.receiveShadow = true;
        m.vertexColors = true; m.metalness = 0;
        if (m.map) m.map.anisotropy = renderer.capabilities.getMaxAnisotropy();
      } else if (/^(trunks|trilithon|door_stone|altar|monolith|rune_stone|small_stone|rock_|rune_pillar)/.test(n)) {
        o.castShadow = true; o.receiveShadow = true;
      } else if (n === 'leaves') {
        o.castShadow = true;   // shadows follow the leaf shapes, not the cards they are painted on
        o.customDepthMaterial = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking, map: m.map, alphaTest: 0.45 });
      }
      if (n === 'moon') { m.fog = false; m.emissiveIntensity = 1.0; }
      if (n === 'mountains') {             // far peaks dissolve into the sky haze instead of standing grey
        m.color = srgb(0xc8d8e8); m.emissive = srgb(0x7e98b0); m.emissiveIntensity = 0.55; o.castShadow = false;
      }
      if (m.name.indexOf('rune_') === 0 && fx.runes.indexOf(m) < 0) fx.runes.push(m);
      if (m.name === 'portal') fx.portal = m;
      var mem = /^memory_(\d)$/.exec(m.name);
      if (mem) fx.memory[parseInt(mem[1], 10) - 1] = m;
    });
    scene.add(world);
    forestScene = world; forestObjs.push(world);
    world.traverse(function (o) { if (!doorNode && /^door_stone/.test(o.name)) doorNode = o; });
    // Each memory stone gets a warm light and a faint beacon so it can be found from afar.
    POINTS.forEach(function (pt) {
      var mem = /^memory_(\d)$/.exec(pt.id);
      if (!mem) return;
      var light = new THREE.PointLight(srgb(0xffa040), 1.4, 16, 2);
      light.position.set(pt.x, pt.y + 2.2, pt.z);
      var beam = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.34, 46, 10, 1, true),
        new THREE.MeshBasicMaterial({ color: srgb(0xffb060), transparent: true, opacity: 0.3, depthWrite: false, fog: false, side: THREE.DoubleSide }));
      beam.position.set(pt.x, pt.y + 25, pt.z);
      scene.add(light, beam);
      forestObjs.push(light, beam);
      fx.beacons[parseInt(mem[1], 10) - 1] = { light: light, beam: beam };
    });
  }

  // ---------------------------------------------------------------- the forest's plants
  // Trees are separate meshes so each can be culled on its own; grass, flowers, ferns, mushrooms
  // and pebbles are instanced in 24 m squares that switch off beyond their draw distance.
  var FOREST = { chunks: [], wind: { value: 0 } }, HERO_GLTF = null, NPCS = [];
  function addWind(material, strength, from) {
    if (material.userData.wind) return;
    material.userData.wind = true;
    material.onBeforeCompile = function (sh) {
      sh.uniforms.uTime = FOREST.wind;
      sh.vertexShader = 'uniform float uTime;\n' + sh.vertexShader.replace('#include <begin_vertex>', [
        '#include <begin_vertex>',
        '#ifdef USE_INSTANCING',
        '  vec3 swayAt = vec3(instanceMatrix[3][0], instanceMatrix[3][1], instanceMatrix[3][2]);',
        '#else',
        '  vec3 swayAt = vec3(modelMatrix[3][0], modelMatrix[3][1], modelMatrix[3][2]);',
        '#endif',
        '  float sway = sin(uTime * 1.6 + swayAt.x * 0.37 + swayAt.z * 0.29) + 0.45 * sin(uTime * 2.9 + swayAt.x * 0.9 - swayAt.z * 0.5);',
        '  float bend = max(position.y - ' + from.toFixed(2) + ', 0.0);',
        '  transformed.x += sway * ' + strength.toFixed(4) + ' * bend * bend;',
        '  transformed.z += sway * ' + (strength * 0.6).toFixed(4) + ' * bend * bend;'].join('\n'));
    };
  }
  function primitivesOf(node) {
    var list = [];
    node.traverse(function (o) { if (o.isMesh) list.push(o); });
    return list;
  }
  function tuneMaterial(m) {
    m.vertexColors = true; m.metalness = 0;
    if (/^(grass|plant)$/.test(m.name)) { m.side = THREE.DoubleSide; addWind(m, 0.16, 0.0); }
    if (m.name === 'leaf') { m.side = THREE.DoubleSide; addWind(m, 0.0016, 3.0); }
    if (/^glow_/.test(m.name)) m.emissiveIntensity = 1.0;
  }
  function buildForest(gltf, inst, data) {
    var protos = {}, drop = [];
    gltf.scene.traverse(function (o) {
      // a multi-material prototype is a group of meshes named proto_x_1, proto_x_2?? take the group
      if (/^proto_/.test(o.name) && !(o.parent && /^proto_/.test(o.parent.name)) && !protos[o.name]) { protos[o.name] = o; drop.push(o); }
    });
    drop.forEach(function (o) { if (o.parent) o.parent.remove(o); });
    var group = new THREE.Group();
    group.name = 'forest_plants';
    var mtx = new THREE.Matrix4(), col = new THREE.Color(), pos = new THREE.Vector3(), size = data.chunk || 24;
    // The ruins share the land's photoscanned plants: keep each old plant's place, pick its real-world counterpart.
    var swap = {
      tree: function (m, i) { return /giant/.test(m) ? 'tree_oak' : /green_b/.test(m) ? 'tree_ash' : /green/.test(m) ? (i % 2 ? 'tree_oak_b' : 'tree_ash') : (i % 3 ? 'tree_fir_b' : 'tree_fir'); },
      fern: function (m, i) { return 'fern_02_' + (i % 4); },
      curl: function (m, i) { return 'shrub_03_' + (i % 4); },
      flower: function (m, i) { return 'dandelion_01_' + (i % 5); },
      mush: function (m, i) { return ['tree_stump_01', 'tree_stump_02', 'root_cluster_01', 'dry_branches_medium_01_0', 'rock_moss_set_01_' + (i % 6), 'nettle_plant_' + (i % 6)][i % 6]; },
      pebble: function (m, i) { return i % 2 ? 'stone_01' : 'rock_09'; }
    };
    FOREST.scatter = {};
    var pp = new THREE.Vector3(), pq = new THREE.Quaternion(), ps = new THREE.Vector3(), pe = new THREE.Euler();
    data.instances.forEach(function (gi) {
      var f = swap[gi.set];
      if (!f) return;                       // grass and blades: the runtime tuft field covers them
      for (var i = 0; i < gi.count; i++) {
        if (gi.set === 'flower' && i % 3) continue;
        var b = (gi.offset + i) * 15;
        mtx.set(inst[b], inst[b + 1], inst[b + 2], inst[b + 3], inst[b + 4], inst[b + 5], inst[b + 6], inst[b + 7],
                inst[b + 8], inst[b + 9], inst[b + 10], inst[b + 11], 0, 0, 0, 1);
        mtx.decompose(pp, pq, ps); pe.setFromQuaternion(pq, 'YXZ');
        var name = f(gi.mesh, i), s = gi.set === 'tree' ? 0.85 + (i % 5) * 0.08 : gi.set === 'pebble' ? 1.6 : gi.set === 'mush' ? 0.8 : 1.0;
        (FOREST.scatter[name] = FOREST.scatter[name] || []).push([pp.x, pp.y, pp.z, pe.y, s]);
      }
    });
    data.instances = [];
    data.instances.forEach(function (gi) {
      var node = protos[gi.mesh];
      if (!node) return;
      var prims = primitivesOf(node);
      prims.forEach(function (p) { tuneMaterial(p.material); if (!p.geometry.boundingSphere) p.geometry.computeBoundingSphere(); });
      function read(i) {
        var b = (gi.offset + i) * 15;
        mtx.set(inst[b], inst[b + 1], inst[b + 2], inst[b + 3], inst[b + 4], inst[b + 5], inst[b + 6], inst[b + 7],
                inst[b + 8], inst[b + 9], inst[b + 10], inst[b + 11], 0, 0, 0, 1);
        col.setRGB(inst[b + 12], inst[b + 13], inst[b + 14]);
      }
      if (gi.draw === 'each') {
        for (var i = 0; i < gi.count; i++) {
          read(i);
          prims.forEach(function (p) {
            var m = new THREE.Mesh(p.geometry, p.material);
            m.matrixAutoUpdate = false; m.matrix.copy(mtx); m.matrixWorldNeedsUpdate = true;
            m.castShadow = true; m.receiveShadow = true;
            group.add(m);
          });
        }
        return;
      }
      var buckets = {};
      for (var k = 0; k < gi.count; k++) {
        var b = (gi.offset + k) * 15;
        var key = Math.floor(inst[b + 3] / size) + ',' + Math.floor(inst[b + 11] / size);
        (buckets[key] = buckets[key] || []).push(k);
      }
      Object.keys(buckets).forEach(function (key) {
        var ids = buckets[key], centre = new THREE.Vector3(), reach = 0;
        ids.forEach(function (k) { var b = (gi.offset + k) * 15; centre.x += inst[b + 3]; centre.y += inst[b + 7]; centre.z += inst[b + 11]; });
        centre.multiplyScalar(1 / ids.length);
        ids.forEach(function (k) { var b = (gi.offset + k) * 15; reach = Math.max(reach, Math.hypot(inst[b + 3] - centre.x, inst[b + 7] - centre.y, inst[b + 11] - centre.z)); });
        prims.forEach(function (p) {
          var geo = new THREE.BufferGeometry();         // shares the prototype's buffers, owns its own bounds
          for (var name in p.geometry.attributes) geo.setAttribute(name, p.geometry.attributes[name]);
          geo.setIndex(p.geometry.getIndex());
          geo.boundingSphere = new THREE.Sphere(centre.clone(), reach + p.geometry.boundingSphere.radius * 1.8);
          var im = new THREE.InstancedMesh(geo, p.material, ids.length);
          ids.forEach(function (k, j) { read(k); im.setMatrixAt(j, mtx); im.setColorAt(j, col); });
          im.instanceMatrix.needsUpdate = true;
          if (im.instanceColor) im.instanceColor.needsUpdate = true;
          im.castShadow = gi.set === 'pebble' || gi.set === 'mush';
          im.receiveShadow = true;
          group.add(im);
          FOREST.chunks.push({ mesh: im, x: centre.x, z: centre.z, far: gi.far + reach });
        });
      });
    });
    gltf.scene.add(group);
    FOREST.group = group;
    if (LAND.protos) buildForestLand(LAND.protos, LAND.gt);     // the land's assets arrived first
  }
  var cullTick = 0;
  function cullForest() {
    if (zone !== 'forest' || (cullTick++ % 6)) return;
    for (var i = 0; i < FOREST.chunks.length; i++) {
      var c = FOREST.chunks[i];
      c.mesh.visible = Math.hypot(camera.position.x - c.x, camera.position.z - c.z) < c.far;
    }
  }

  function setupHero(gltf) {
    HERO_REST = {};                       // the rest pose, for dressing villagers before anything animates
    gltf.scene.traverse(function (o) { if (o.isBone) HERO_REST[o.name] = [o.position.clone(), o.quaternion.clone(), o.scale.clone()]; });
    hero.root = gltf.scene;
    hero.root.traverse(function (o) {
      if (!o.isMesh) return;
      o.castShadow = true;
      o.frustumCulled = false;   // the rest-pose bounds do not cover lying down or a flip
      o.material.side = THREE.DoubleSide;
      o.material.metalness = 0;
    });
    scene.add(hero.root);
    hero.mixer = new THREE.AnimationMixer(hero.root);
    gltf.animations.forEach(function (clip) {
      var action = hero.mixer.clipAction(clip);
      if (['Jump', 'DoubleJump', 'Attack', 'WakeUp', 'Slash'].indexOf(clip.name) >= 0) {
        action.setLoop(THREE.LoopOnce, 1);
        action.clampWhenFinished = true;
      }
      hero.actions[clip.name] = action;
    });
    hero.mixer.addEventListener('finished', function (e) {
      var name = e.action.getClip().name;
      if (name === 'Attack' || name === 'Slash') hero.attacking = false;
      if (name === 'DoubleJump') hero.flipping = false;
    });
  }

  // ---------------------------------------------------------------- story
  var intro = { woke: false, ending: false };
  var doorFx = { t: -1, glow: null, light: null, shake: 0 };
  function glowTexture() {
    var c = document.createElement('canvas'); c.width = c.height = 128;
    var g = c.getContext('2d'), grd = g.createRadialGradient(64, 64, 0, 64, 64, 64);
    grd.addColorStop(0, 'rgba(255,255,255,1)'); grd.addColorStop(0.25, 'rgba(200,245,255,0.85)');
    grd.addColorStop(0.6, 'rgba(120,220,255,0.25)'); grd.addColorStop(1, 'rgba(120,220,255,0)');
    g.fillStyle = grd; g.fillRect(0, 0, 128, 128);
    return new THREE.CanvasTexture(c);
  }
  function startDoorFx() {
    if (doorFx.t >= 0) return;
    var p = new THREE.Vector3(), q = new THREE.Quaternion(), sc = new THREE.Vector3();
    if (doorNode) { doorNode.updateWorldMatrix(true, false); doorNode.matrixWorld.decompose(p, q, sc); }
    else { var d = POINTS.filter(function (x) { return x.id === 'door'; })[0]; p.set(d.x, d.y, d.z); }
    var out = new THREE.Vector3(0, 0, 1).applyQuaternion(q);
    var at = p.clone().add(out.multiplyScalar(0.5)); at.y += doorNode ? sc.y * 0.42 : 1.8;
    doorFx.glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTexture(), color: 0xffffff, transparent: true, opacity: 0,
      depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending, fog: false }));
    doorFx.glow.position.copy(at); doorFx.glow.renderOrder = 10;
    doorFx.light = new THREE.PointLight(srgb(0x9fe8ff), 0, 30, 2); doorFx.light.position.copy(at);
    scene.add(doorFx.glow, doorFx.light); forestObjs.push(doorFx.glow, doorFx.light);
    doorFx.t = 0;
  }
  function onStage(name) {
    if (name === 'door_open') { startDoorFx(); fx.pulse = 3; }
    if (name === 'summon') { doorFx.t = Math.max(doorFx.t, 2.2); fx.pulse = 3; }
    if (name === 'open_eyes') $('veil').classList.add('open');
    if ((name === 'wake' || name === 'standing') && !intro.woke) { intro.woke = true; play('WakeUp', 0.5); cam.goal = 4.4; }
    if (name === 'pulse') fx.pulse = 2.8;
  }
  function setObjective() {
    if ($('questlist')) $('questlist').hidden = true;
    if (zone === 'village') { $('objective').textContent = questObjective(); renderQuestList(); return; }
    var o = STORY.objective;
    $('objective').textContent = finished ? '메갈리스의 문이 열리려 한다' : foundCount === 0 ? o.start : foundCount < 3 ? o.progress.replace('{n}', foundCount) : o.done;
  }
  function startIntro() {
    play('Sleep', 0);
    say(STORY.intro, function () { intro.ending = true; hero.locked = true; });
  }
  function finishIntro() {
    intro.ending = false;
    hero.cutscene = false; hero.locked = false;
    play('Idle', 0.4);
    cam.goal = 5.6;
    setObjective();
    $('nameplate').hidden = false;
    $('legend').hidden = false;
  }
  function interact(pt) {
    var entry = STORY.interact[pt.id], mem = /^memory_(\d)$/.exec(pt.id);
    hero.vel.x = hero.vel.z = 0;
    if (mem) {
      var i = parseInt(mem[1], 10) - 1;
      if (found[i]) { say(entry.lines_again); return; }
      fx.pulse = 1.5;
      say(entry.lines, function () {
        found[i] = true; foundCount += 1; setObjective();
        if (foundCount === 3) say(STORY.events.all_memories);
      });
    } else if (pt.id === 'door' && foundCount === 3) {
      fx.pulse = 3;
      say(entry.lines_done, function () { finished = true; summon(); });
    } else if (pt.npc) {
      talkTo(pt.npc);
    } else say(entry.lines);
  }
  var nearest = null;
  function updatePrompt() {
    nearest = null;
    if (!hero.locked && hero.onFloor) {
      var best = 1e9;
      for (var i = 0; i < POINTS.length; i++) {
        var d = Math.hypot(hero.pos.x - POINTS[i].x, hero.pos.z - POINTS[i].z);
        if (d < POINTS[i].radius && d < best && Math.abs(hero.pos.y - POINTS[i].y) < 2.5) { best = d; nearest = POINTS[i]; }
      }
      if (zone === 'village') {
        for (var k = 0; k < NPCS.length; k++) {
          var dn = Math.hypot(hero.pos.x - NPCS[k].pos.x, hero.pos.z - NPCS[k].pos.z);
          if (dn < 2.8 && dn < best + 1.0) { best = dn; nearest = { npc: NPCS[k], id: NPCS[k].id }; }
        }
      }
    }
    var el = $('prompt');
    if (!nearest) { if (!el.hidden) el.hidden = true; return; }
    var name = nearest.npc ? nearest.npc.def.name : STORY.interact[nearest.id].name;
    var verb = nearest.npc ? particle(name) + ' 이야기하기' : ' 살펴보기';
    if (el.hidden || $('prompt-name').textContent !== name) { $('prompt-name').textContent = name; $('prompt-verb').textContent = verb; el.hidden = false; }
  }

  // ---------------------------------------------------------------- simulation
  function approach(value, target, step) { return value < target ? Math.min(value + step, target) : Math.max(value - step, target); }

  function step(dt) {
    var p = hero.pos, v = hero.vel;
    if (hero.locked && !talk.active && !intro.ending && !hero.cutscene && clockNow >= unlockAt) hero.locked = false;
    var free = !hero.locked;
    var ix = free ? (keys.right ? 1 : 0) - (keys.left ? 1 : 0) + stickVec.x : 0;
    var iy = free ? (keys.forward ? 1 : 0) - (keys.back ? 1 : 0) - stickVec.y : 0;
    var len = Math.hypot(ix, iy), dirX = 0, dirZ = 0;
    if (len > 0.12) {
      ix /= Math.max(len, 1); iy /= Math.max(len, 1);
      var fwx = -Math.sin(cam.yaw), fwz = -Math.cos(cam.yaw), rx = Math.cos(cam.yaw), rz = -Math.sin(cam.yaw);
      dirX = rx * ix + fwx * iy; dirZ = rz * ix + fwz * iy;
      var dl = Math.hypot(dirX, dirZ); dirX /= dl; dirZ /= dl;
    }
    var speed = RUN_SPEED;          // Robin always runs: there is no walking pace to hold a key for
    if (hero.attacking && hero.onFloor) speed = 0;
    var accel = (hero.onFloor ? ACCEL : AIR_ACCEL) * dt;
    v.x = approach(v.x, dirX * speed, accel);
    v.z = approach(v.z, dirZ * speed, accel);

    if (free) {
      if (pressed.interact && nearest) interact(nearest);
      else if (pressed.jump) {
        if (hero.onFloor) { hero.jumpsUsed = 1; v.y = JUMP_VELOCITY; hero.onFloor = false; hero.attacking = false; replay('Jump', 0.08); }
        else if (hero.jumpsUsed < MAX_JUMPS) { hero.jumpsUsed = MAX_JUMPS; v.y = DOUBLE_JUMP_VELOCITY; hero.flipping = true; replay('DoubleJump', 0.05); }
      } else if (pressed.attack && hero.onFloor && !hero.attacking) { hero.attacking = true;
        if (INV.held && hero.actions.Slash) { replay('Slash', 0.08, 1.15); COMBAT.strike = 0.26; } else { replay('Attack', BLEND, ATTACK_SPEED); COMBAT.strike = 0.24; } }
    }
    pressed.jump = pressed.attack = pressed.interact = false;

    if ((dirX || dirZ) && !(hero.attacking && hero.onFloor)) {
      var want = Math.atan2(dirX, dirZ), diff = Math.atan2(Math.sin(want - hero.heading), Math.cos(want - hero.heading));
      hero.heading += diff * Math.min(1, TURN_SPEED * dt);
    }

    // Sideways first (stones, trunks, the world's edge), then vertical (ground and stone tops).
    var prevX = p.x, prevZ = p.z;
    p.x += v.x * dt; p.z += v.z * dt;
    pushOut(p, v);
    if (zone === 'village' && LAND.ready && floorAt(p.x, p.z, p.y) < LAND.water - 0.7) { p.x = prevX; p.z = prevZ; }   // the sea stops Robin at wading depth (a bridge deck is floor)
    var bc = zone === 'village' ? VILLAGE.bounds : { x: 0, z: 0, r: WORLD.bounds };
    var r = Math.hypot(p.x - bc.x, p.z - bc.z);
    if (r > bc.r) {
      p.x = bc.x + (p.x - bc.x) * bc.r / r; p.z = bc.z + (p.z - bc.z) * bc.r / r;
      if (free && clockNow - flags.edgeAt > 20) { flags.edgeAt = clockNow; v.x = v.z = 0; say(zone === 'village' ? STORY.village.edge : STORY.events.world_edge); }
    }
    if (zone === 'forest' && free && !flags.left && r > WORLD.plateau_radius + 7) { flags.left = true; say(STORY.events.leave_circle); }

    var floor = floorAt(p.x, p.z, p.y);
    if (hero.onFloor) {
      if (p.y - floor <= 0.5) { p.y = floor; v.y = 0; }      // follow the slope
      else hero.onFloor = false;                               // stepped off a stone
    } else {
      v.y -= GRAVITY * dt;
      p.y += v.y * dt;
      if (v.y <= 0 && p.y <= floor) { p.y = floor; v.y = 0; hero.onFloor = true; hero.jumpsUsed = 0; hero.flipping = false; }
    }

    if (!hero.cutscene && !hero.attacking && !hero.flipping) {
      var ground = Math.hypot(v.x, v.z);
      if (!hero.onFloor) {
        var rising = hero.current === 'Jump' && hero.actions.Jump.isRunning();
        if (v.y <= 0.5 || !rising) play('Fall', BLEND);
      } else if (ground > (WALK_SPEED + RUN_SPEED) / 2) play('Run', BLEND, Math.max(0.7, Math.min(1.6, ground / RUN_CLIP_SPEED)));
      else if (ground > 0.2) play('Walk', BLEND, Math.max(0.6, Math.min(1.6, ground / WALK_CLIP_SPEED)));
      else play('Idle', BLEND);
    }
  }

  function updateEffects(dt) {
    fx.pulse = Math.max(0, fx.pulse - dt);
    var breathe = 0.5 + 0.5 * Math.sin(clockNow * 1.7);
    var boost; boost = fx.pulse > 0 ? 1.4 * Math.sin(Math.min(1, fx.pulse) * Math.PI * 0.5) * (0.7 + 0.3 * Math.sin(clockNow * 9)) : 0;
    for (var i = 0; i < fx.runes.length; i++) fx.runes[i].emissiveIntensity = 0.34 + 0.14 * breathe + boost * 0.8;
    for (var k = 0; k < 3; k++) {
      if (fx.memory[k]) fx.memory[k].emissiveIntensity = found[k] ? 1.2 : 0.3 + 0.7 * (0.5 + 0.5 * Math.sin(clockNow * 2.6 + k * 2));
      if (fx.beacons[k]) {
        fx.beacons[k].beam.material.opacity = found[k] ? 0.07 : 0.2 + 0.12 * breathe;
        fx.beacons[k].light.intensity = found[k] ? 0.9 : 1.0 + 0.8 * breathe;
      }
    }
    if (doorFx.t >= 0) {
      doorFx.t += dt;
      var k = Math.min(1, doorFx.t / 3.2), k2 = k * k;
      doorFx.glow.material.opacity = Math.min(1, 0.2 + 1.1 * k);
      doorFx.glow.scale.setScalar(1.2 + 9 * k2 + 0.25 * Math.sin(clockNow * 7));
      doorFx.light.intensity = 9 * k;
      doorFx.shake = 0.035 * k2;
      boost += 4 * k;
    }
    if (VILLAGE && VILLAGE.water) VILLAGE.water.uTime.value = clockNow;
    if (VILLAGE && VILLAGE.wheel) VILLAGE.wheel.rotation.x -= dt * 0.6;
    if (fx.portal) fx.portal.emissiveIntensity = (foundCount === 3 ? 1.1 : 0.6) + (foundCount === 3 ? 0.4 : 0.15) * Math.sin(clockNow * 2.2) + boost * 0.5;
  }

  // ---------------------------------------------------------------- frame loop
  var focus = new THREE.Vector3(hero.pos.x, hero.pos.y + 0.4, hero.pos.z), want = new THREE.Vector3();

  function resize() {
    var w = stage.clientWidth, h = stage.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  window.addEventListener('resize', resize);
  if (window.ResizeObserver) new ResizeObserver(resize).observe(stage);
  resize();

  function update(dt) {
    clockNow += dt;
    cam.yaw += ((keys.camLeft ? 1 : 0) - (keys.camRight ? 1 : 0)) * 2.2 * dt;
    cam.distance += (cam.goal - cam.distance) * Math.min(1, 2.5 * dt);
    updateTalk(dt);
    if (hero.root) {
      step(dt);
      hero.mixer.update(dt);
      if (intro.ending && !hero.actions.WakeUp.isRunning()) finishIntro();
      hero.root.position.copy(hero.pos);
      hero.root.rotation.y = hero.heading;
      updatePrompt();
    }
    updateEffects(dt);
    updateNpcs(dt);
    FOREST.wind.value = clockNow;
    var lying = hero.cutscene && !intro.woke;
    want.set(hero.pos.x, hero.pos.y + (lying ? 0.45 : 1.3), hero.pos.z);
    focus.lerp(want, Math.min(1, (lying ? 3 : 8) * dt));
    var cp = Math.cos(cam.pitch);
    // indoors the camera comes in close (and eases back out at the door)
    var inside = zone === 'village' && roomAt(hero.pos.x, hero.pos.y + 0.5, hero.pos.z, -0.05);
    cam.indoor += ((inside ? 1 : 0) - (cam.indoor || 0)) * Math.min(1, 4 * dt);
    var dist = cam.distance + (Math.min(cam.distance, 3.0) - cam.distance) * (cam.indoor || 0);
    var cx = focus.x + Math.sin(cam.yaw) * cp * dist, cz = focus.z + Math.cos(cam.yaw) * cp * dist;
    camera.position.set(cx, Math.max(groundAt(cx, cz) + 0.35, focus.y + Math.sin(cam.pitch) * dist), cz);
    clampCameraToRoom();
    updateRooms();
    camera.lookAt(focus);
    if (doorFx.shake && zone === 'forest') { camera.position.x += (Math.random() - 0.5) * doorFx.shake; camera.position.y += (Math.random() - 0.5) * doorFx.shake; }
    sky.position.copy(camera.position);
    if (LAND.skyMesh) LAND.skyMesh.position.copy(camera.position);
    sun.target.position.copy(focus);
    sun.position.copy(focus).add(SUN_OFFSET);
    cullForest();
    updateLand();
    updateMonsters(dt);
    updateMusic(dt);
    updateArea();
  }

  // ---------------------------------------------------------------- music by place
  // ruins: Weight of Ancient Stone; village: The Lantern Maker's Toll; the wild beyond: Where the Map Ends.
  // Each track loops from the start; moving between places cross-fades over about two seconds.
  var MUSIC = { tracks: {}, want: null, on: false, level: 0.55 };
  ['ruins', 'village', 'wild'].forEach(function (k) {
    var a = new Audio('music/' + k + '.mp3');
    a.loop = true; a.preload = 'auto'; a.volume = 0;
    MUSIC.tracks[k] = a;
  });
  function startMusic() { MUSIC.on = true; }          // called from a click: browsers only allow sound after one
  function updateMusic(dt) {
    if (!MUSIC.on) return;
    var want = zone === 'forest' ? 'ruins' : (inVillage(hero.pos.x, hero.pos.z, 3) ? 'village' : 'wild');
    Object.keys(MUSIC.tracks).forEach(function (k) {
      var a = MUSIC.tracks[k], target = k === want ? MUSIC.level : 0;
      a.volume = Math.max(0, Math.min(1, a.volume + Math.sign(target - a.volume) * Math.min(Math.abs(target - a.volume), dt * 0.3)));
      if (k === want && a.paused) { if (a.ended || a.volume === 0) a.currentTime = a.currentTime || 0; a.play().catch(function () {}); }
      if (k !== want && !a.paused && a.volume < 0.01) { a.pause(); a.currentTime = 0; }
    });
  }

  var clock = new THREE.Clock();
  // Shadows are redrawn every other frame (the sun barely moves between two frames), and the render
  // resolution adapts only when the frame rate falls: full sharpness whenever the GPU keeps up.
  // Full resolution and fresh shadows every frame; speed comes from culling, LOD and precompiled shaders.
  function frame() {
    requestAnimationFrame(frame);
    update(Math.min(clock.getDelta(), 0.05));
    renderer.render(scene, camera);
  }

  // ---------------------------------------------------------------- the village Robin is summoned to
  var villageWait = [];
  function whenVillage(fn) { if (VILLAGE && VILLAGE.ready) fn(); else villageWait.push(fn); }
  function toFloat(attr, n) {
    var cnt = attr.count, out = new Float32Array(cnt * n), arr = attr.isInterleavedBufferAttribute ? attr.data.array : attr.array;
    var div = !attr.normalized ? 1 : arr instanceof Int16Array ? 32767 : arr instanceof Int8Array ? 127 : arr instanceof Uint8Array ? 255 : arr instanceof Uint16Array ? 65535 : 1;
    var get = [attr.getX, attr.getY, attr.getZ, attr.getW];
    for (var i = 0; i < cnt; i++) for (var c = 0; c < n; c++) {
      var v = get[c].call(attr, i) / div;
      out[i * n + c] = attr.normalized ? Math.max(-1, v) : v;
    }
    return out;
  }
  function linearColors(geo) {
    var col = geo.getAttribute('color');
    if (!col) return;
    var f = toFloat(col, 3);
    for (var i = 0; i < f.length; i++) f[i] = Math.pow(Math.max(0, f[i]), 2.2);
    geo.setAttribute('color', new THREE.BufferAttribute(f, 3));
  }
  function bakedGeometry(mesh) {
    // Dequantise a mesh and fold its node transform into the vertices (prototypes, the sea).
    mesh.updateWorldMatrix(true, false);
    var src = mesh.geometry, g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(toFloat(src.getAttribute('position'), 3), 3));
    if (src.getAttribute('normal')) g.setAttribute('normal', new THREE.BufferAttribute(toFloat(src.getAttribute('normal'), 3), 3));
    if (src.getAttribute('color')) { g.setAttribute('color', src.getAttribute('color')); linearColors(g); }
    g.setIndex(src.getIndex());
    g.applyMatrix4(mesh.matrixWorld);
    if (g.getAttribute('normal')) g.getAttribute('normal').needsUpdate = true;
    g.computeBoundingSphere();
    return g;
  }
  function waterMaterial() {
    var m = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.18, metalness: 0.0 });
    var uniforms = { uTime: { value: 0 } };
    m.onBeforeCompile = function (sh) {
      sh.uniforms.uTime = uniforms.uTime;
      sh.vertexShader = 'uniform float uTime;\n' + sh.vertexShader
        .replace('#include <beginnormal_vertex>', [
          'float wx = position.x, wz = position.z;',
          'float d1 = 0.35 * 0.07 * cos(wx * 0.35 + uTime * 1.1) + 0.9 * 0.025 * cos((wx + wz) * 0.9 + uTime * 1.9);',
          'float d2 = -0.27 * 0.06 * cos(wz * 0.27 - uTime * 0.8) + 0.9 * 0.025 * cos((wx + wz) * 0.9 + uTime * 1.9) + 1.7 * 0.012 * cos(wz * 1.7 + wx * 0.4 + uTime * 2.6);',
          'vec3 objectNormal = normalize(vec3(-d1, 1.0, -d2));',
          '#ifdef USE_TANGENT', 'vec3 objectTangent = vec3(tangent.xyz);', '#endif'].join('\n'))
        .replace('#include <begin_vertex>', [
          'vec3 transformed = vec3(position);',
          'transformed.y += 0.07 * sin(wx * 0.35 + uTime * 1.1) + 0.06 * sin(wz * 0.27 - uTime * 0.8) + 0.025 * sin((wx + wz) * 0.9 + uTime * 1.9) + 0.012 * sin(wz * 1.7 + wx * 0.4 + uTime * 2.6);'].join('\n'));
    };
    return { material: m, uniforms: uniforms };
  }
  function inVillage(x, z, margin) {
    // inside the town wall (closed along the shore); `margin` widens it outward
    var W = VILLAGE && VILLAGE.wall;
    if (!W) return Math.hypot(x, z - 3) < 38 + (margin || 0);
    var c = false, i, j;
    for (i = 0, j = W.length - 1; i < W.length; j = i++) {
      var xi = W[i][0], zi = W[i][1], xj = W[j][0], zj = W[j][1];
      if ((zi > z) !== (zj > z) && x < (xj - xi) * (z - zi) / (zj - zi) + xi) c = !c;
    }
    if (c || !margin) return c;
    for (i = 0, j = W.length - 1; i < W.length; j = i++) {
      var ax = W[j][0], az = W[j][1], dx = W[i][0] - ax, dz = W[i][1] - az;
      var t = Math.max(0, Math.min(1, ((x - ax) * dx + (z - az) * dz) / (dx * dx + dz * dz)));
      if (Math.hypot(x - ax - t * dx, z - az - t * dz) < margin) return true;
    }
    return false;
  }
  function buildVillageExt(V, protos, ext, xd) {
    // The enlarged village (tools/hq_village.py): new buildings in the village's own materials, the new roofs laid
    // with the village's slate pieces, stones and wall segments as instances, the wall's colliders and outline.
    var group = V.group, mats = {}, kit = {}, add = [];
    group.traverse(function (o) { if (o.isMesh && o.material && !/^terrain/.test(o.name) && !mats[o.material.name]) mats[o.material.name] = o.material; });
    Object.keys(protos).forEach(function (k) { var m = protos[k].material; if (m && !mats[m.name]) mats[m.name] = m; });
    ext.scene.updateMatrixWorld(true);
    ext.scene.traverse(function (o) {
      if (!o.isMesh) return;
      if (/^kit_/.test(o.name)) kit[o.name] = o; else add.push(o);
    });
    add.forEach(function (o) {
      var mn = o.material.name, m = mats[mn];
      linearColors(o.geometry);
      if (!m) { m = o.material; m.vertexColors = true; m.metalness = 0; if (mn !== 'glass') detailFor(m); mats[mn] = m; }
      o.material = m;
      o.castShadow = mn !== 'glass'; o.receiveShadow = true;
      var mw = o.matrixWorld.clone();
      group.add(o);
      mw.decompose(o.position, o.quaternion, o.scale);
    });
    var mtx = new THREE.Matrix4(), col = new THREE.Color();
    xd.instances.forEach(function (gi) {
      var src = kit[gi.proto] || protos[gi.proto];
      if (!src) return;
      var geo = bakedGeometry(src), mat = kit[gi.proto] ? (mats[src.material.name] || src.material) : src.material;
      mat.vertexColors = true;
      var bin = atob(gi.f32), buf = new ArrayBuffer(bin.length), u8 = new Uint8Array(buf);
      for (var i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
      var f = new Float32Array(buf);
      var im = new THREE.InstancedMesh(geo, mat, gi.count);
      for (var k = 0; k < gi.count; k++) {
        var b = k * 15;
        mtx.set(f[b], f[b + 1], f[b + 2], f[b + 3], f[b + 4], f[b + 5], f[b + 6], f[b + 7], f[b + 8], f[b + 9], f[b + 10], f[b + 11], 0, 0, 0, 1);
        im.setMatrixAt(k, mtx);
        col.setRGB(f[b + 12], f[b + 13], f[b + 14]);
        im.setColorAt(k, col);
      }
      im.instanceMatrix.needsUpdate = true;
      if (im.instanceColor) im.instanceColor.needsUpdate = true;
      im.frustumCulled = false; im.castShadow = true; im.receiveShadow = true;
      im.name = 'ext_' + gi.proto;
      group.add(im);
    });
    V.boxes = V.boxes.concat(boxesOf(xd.colliders));
    V.wall = xd.wall;
    V.gates = xd.gates;
    // rooms you can walk into: their bounds (for the camera) and their lamps (fire and candlelight)
    V.rooms = (xd.interiors || []).map(function (r) {
      var room = { name: r.name, id: r.id, x: r.x, z: r.z, cos: Math.cos(r.yaw), sin: Math.sin(r.yaw), hx: r.hx, hz: r.hz, floor: r.floor, ceil: r.ceil, lights: [] };
      r.lights.forEach(function (l, i) {
        var pl = new THREE.PointLight(srgb(new THREE.Color(l.color[0], l.color[1], l.color[2]).getHex()), 0, l.distance, 2);
        pl.position.set(l.x, l.y, l.z);
        group.add(pl);
        room.lights.push({ light: pl, base: l.intensity, seed: i * 1.7 + r.x });
      });
      return room;
    });
  }
  function roomAt(x, y, z, margin) {
    var rooms = VILLAGE && VILLAGE.rooms;
    if (!rooms) return null;
    for (var i = 0; i < rooms.length; i++) {
      var r = rooms[i], l = toLocal(r, x, z);
      if (Math.abs(l[0]) < r.hx + margin && Math.abs(l[1]) < r.hz + margin && y > r.floor - 0.6 && y < r.ceil) return r;
    }
    return null;
  }
  function roomFloorAt(x, z, margin) {
    var rooms = VILLAGE && VILLAGE.rooms;
    if (!rooms) return false;
    for (var i = 0; i < rooms.length; i++) {
      var r = rooms[i], l = toLocal(r, x, z);
      if (Math.abs(l[0]) < r.hx + margin && Math.abs(l[1]) < r.hz + margin) return true;
    }
    return false;
  }
  function updateRooms() {
    // lamps flicker, and only burn while Robin is near enough to see them (lights far off cost for nothing)
    if (zone !== 'village' || !VILLAGE || !VILLAGE.rooms) return;
    VILLAGE.rooms.forEach(function (r) {
      var near = Math.hypot(hero.pos.x - r.x, hero.pos.z - r.z) < 40;
      r.lights.forEach(function (L) {
        var f = 0.86 + 0.09 * Math.sin(clockNow * 11.3 + L.seed) + 0.05 * Math.sin(clockNow * 27.1 + L.seed * 2.3);
        L.light.intensity = near ? L.base * f : 0;
      });
    });
  }
  function clampCameraToRoom() {
    // indoors the camera stays inside the walls and under the ceiling
    if (zone !== 'village') return;
    var r = roomAt(hero.pos.x, hero.pos.y + 0.5, hero.pos.z, -0.05);
    if (!r) return;
    var l = toLocal(r, camera.position.x, camera.position.z);
    var lx = Math.max(-r.hx + 0.3, Math.min(r.hx - 0.3, l[0])), lz = Math.max(-r.hz + 0.3, Math.min(r.hz - 0.3, l[1]));
    camera.position.x = r.x + lx * r.cos + lz * r.sin;
    camera.position.z = r.z - lx * r.sin + lz * r.cos;
    camera.position.y = Math.max(r.floor + 0.5, Math.min(r.ceil - 0.25, camera.position.y));
  }
  function buildVillage(gltf, inst, data, ext, extData) {
    var group = gltf.scene, protos = {}, drop = [];
    group.visible = false;
    var maxAniso = renderer.capabilities.getMaxAnisotropy();
    var V = { ready: false, group: group, data: data, water: null, wheel: null };
    group.traverse(function (o) {
      if (!o.isMesh) return;
      if (o.name.indexOf('proto_') === 0) { protos[o.name] = o; drop.push(o); return; }
      var mn = o.material.name;
      if (mn === 'water') {
        var wm = waterMaterial();
        o.geometry = bakedGeometry(o); o.position.set(0, 0, 0); o.quaternion.identity(); o.scale.set(1, 1, 1);
        o.material = wm.material; V.water = wm.uniforms; o.receiveShadow = true; return;
      }
      linearColors(o.geometry);
      o.material.vertexColors = true;
      o.material.metalness = mn === 'iron' ? 0.55 : 0;
      if (mn === 'glass') { o.material.roughness = 0.12; o.material.metalness = 0.35; }
      if (mn === 'cloth' || mn === 'grass') o.material.side = THREE.DoubleSide;
      var big = /^(terrain|castle|sea)/.test(o.name);
      o.castShadow = !big && mn !== 'glass';
      o.receiveShadow = true;
      if (/^castle_/.test(o.name)) { o.visible = false; return; }        // the first castle and its rock island: replaced by the castle out at sea
      if (/^terrain/.test(o.name)) { o.material = o.material.clone(); o.material.userData = { clipR: 27 }; o.material.color.setRGB(0.66, 0.62, 0.64); }   // the land takes over beyond; toned to sit with its scanned meadow
      if (mn !== 'glass') detailFor(o.material, /^castle/.test(o.name) ? { stone: ['ashlar'] } : null);
    });
    drop.forEach(function (o) { o.parent.remove(o); });
    var mtx = new THREE.Matrix4(), col = new THREE.Color();
    data.instances.forEach(function (gi) {
      var pm = protos[gi.mesh];
      if (!pm) return;
      var geo = bakedGeometry(pm), mat = pm.material;
      if (gi.set === 'grass') { mat = mat.clone(); mat.userData = {}; mat.color.setRGB(0.62, 0.6, 0.55); }   // toned to the scanned meadow round it
      mat.vertexColors = true; mat.metalness = 0;
      if (gi.set === 'grass') mat.side = THREE.DoubleSide;
      else detailFor(mat);
      var im = new THREE.InstancedMesh(geo, mat, gi.count);
      for (var i = 0; i < gi.count; i++) {
        var b = (gi.offset + i) * 15;
        mtx.set(inst[b], inst[b + 1], inst[b + 2], inst[b + 3], inst[b + 4], inst[b + 5], inst[b + 6], inst[b + 7],
                inst[b + 8], inst[b + 9], inst[b + 10], inst[b + 11], 0, 0, 0, 1);
        im.setMatrixAt(i, mtx);
        col.setRGB(inst[b + 12], inst[b + 13], inst[b + 14]);
        im.setColorAt(i, col);
      }
      im.instanceMatrix.needsUpdate = true;
      if (im.instanceColor) im.instanceColor.needsUpdate = true;
      im.frustumCulled = false;
      im.castShadow = gi.set === 'slate' || gi.set === 'stone' || gi.set === 'ridge';
      im.receiveShadow = true;
      im.name = gi.mesh;
      group.add(im);
    });
    // standing stones out in the sea, cut from the same rock as the ones in the forest
    if (forestScene) {
      var srcs = {};
      forestScene.traverse(function (o) { if (o.isMesh) { var m = /^(trilithon|monolith)/.exec(o.name); if (m && !srcs[m[1]]) srcs[m[1]] = o; } });
      data.sea_stones.forEach(function (st) {
        var src = srcs[st.kind];
        if (!src) return;
        var c = new THREE.Mesh(src.geometry, src.material);
        src.updateWorldMatrix(true, false);
        var p0 = new THREE.Vector3(), q0 = new THREE.Quaternion(), s0 = new THREE.Vector3();
        src.matrixWorld.decompose(p0, q0, s0);
        c.scale.copy(s0).multiplyScalar(st.s);
        c.quaternion.copy(q0).premultiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), st.yaw));
        c.position.set(st.x, data.water_level - 2.2, st.z);
        c.castShadow = false; c.receiveShadow = true;
        group.add(c);
      });
    }
    scene.add(group);
    var H = data.height, bin = atob(H.int16_le_base64), hv = new Float32Array(bin.length / 2);
    var dv = new DataView(new ArrayBuffer(bin.length));
    for (var i = 0; i < bin.length; i++) dv.setUint8(i, bin.charCodeAt(i));
    for (var k = 0; k < hv.length; k++) hv[k] = dv.getInt16(k * 2, true) * H.unit;
    V.groundAt = function (x, z) {
      var fx = Math.max(0, Math.min(H.size - 1.001, (x + H.half) / H.step));
      var fz = Math.max(0, Math.min(H.size - 1.001, (z + H.half) / H.step));
      var ix = Math.floor(fx), iz = Math.floor(fz), tx = fx - ix, tz = fz - iz, row = iz * H.size + ix;
      var a = hv[row] * (1 - tx) + hv[row + 1] * tx, b = hv[row + H.size] * (1 - tx) + hv[row + H.size + 1] * tx;
      return a * (1 - tz) + b * tz;
    };
    V.boxes = boxesOf(data.colliders);
    V.bounds = data.bounds;
    if (ext && extData) {
      try { buildVillageExt(V, protos, ext, extData); } catch (err) { console.error('village ext', err); }
    }
    V.ready = true;
    VILLAGE = V;
    // compile the village shaders now, while nobody is looking, so the summon does not stutter
    group.visible = true; renderer.compile(scene, camera); group.visible = false;
    buildNpcs();
    var w = villageWait; villageWait = [];
    w.forEach(function (fn) { fn(); });
  }
  // ---------------------------------------------------------------- villagers
  // Every villager is Robin's own mesh and rig, re-dressed with a recoloured texture and
  // Blender-made props hung on the bones, so they share his quality and his animations.
  var NPC_PACK = null, HERO_REST = null;
  function particle(name) {           // 와/과 after a Korean name
    var code = name.charCodeAt(name.length - 1) - 0xac00;
    return code >= 0 && code < 11172 && code % 28 ? '과' : '와';
  }
  function buildNpcs() {
    if (!NPC_PACK || !HERO_GLTF || !(VILLAGE && VILLAGE.ready) || NPCS.length || !THREE.SkeletonUtils) return;
    var defs = STORY.village.npcs, skins = {}, faces = {}, props = {};
    NPC_PACK.scene.traverse(function (o) {
      if (o.isMesh && /^skin_/.test(o.material.name)) skins[o.material.name.slice(5)] = o.material.map;
      if (o.isMesh && /^face_/.test(o.material.name)) faces[o.material.name.slice(5)] = o.material.map;
      if (/^prop_/.test(o.name) && !props[o.name.slice(5)]) props[o.name.slice(5)] = o;
    });
    Object.keys(defs).forEach(function (id, index) {
      var d = defs[id];
      var root = THREE.SkeletonUtils.clone(HERO_GLTF.scene);
      root.position.set(0, 0, 0); root.rotation.set(0, 0, 0); root.scale.set(1, 1, 1);
      var bones = {};
      root.traverse(function (o) {
        if (o.isBone) {
          bones[o.name] = o;
          var rest = HERO_REST[o.name];
          if (rest) { o.position.copy(rest[0]); o.quaternion.copy(rest[1]); o.scale.copy(rest[2]); }
        }
        if (o.isMesh) {
          o.castShadow = true; o.receiveShadow = true; o.frustumCulled = false;
          if (o.material.name === 'hero' && skins[d.skin]) { o.material = o.material.clone(); o.material.map = skins[d.skin]; }
          if (o.material.name === 'hero_face' && faces[d.skin]) { o.material = o.material.clone(); o.material.map = faces[d.skin]; }
        }
      });
      root.updateMatrixWorld(true);
      (d.props || []).forEach(function (pr) {
        var src = props[pr[0]];
        if (!src || !bones[pr[1]]) return;
        var piece = src.clone();
        piece.position.set(0, 0, 0); piece.quaternion.identity(); piece.scale.set(1, 1, 1);
        piece.traverse(function (o) { if (o.isMesh) { o.castShadow = true; o.material.vertexColors = true; o.frustumCulled = false; } });
        root.add(piece);
        piece.updateMatrixWorld(true);
        bones[pr[1]].attach(piece);          // keeps the place it was modelled in, then follows the bone
      });
      if (d.scale) root.scale.setScalar(d.scale);
      var mixer = new THREE.AnimationMixer(root), actions = {};
      HERO_GLTF.animations.forEach(function (clip) { if (/^(Idle|Walk|Run)$/.test(clip.name)) actions[clip.name] = mixer.clipAction(clip); });
      var start = d.at || (d.path ? d.path[0] : [d.circle[0], d.circle[1] + d.circle[2]]);
      var n = { id: id, def: d, root: root, bones: bones, mixer: mixer, actions: actions, current: '',
                pos: new THREE.Vector3(start[0], VILLAGE.groundAt(start[0], start[1]), start[1]), heading: 0,
                leg: 1, pause: 0, ang: index * 1.3, talking: false, met: false };
      if (d.face) n.heading = Math.atan2(d.face[0] - start[0], d.face[1] - start[1]);
      npcPlay(n, 'Idle', 1);
      actions.Idle.time = Math.random() * 3;
      VILLAGE.group.add(root);
      NPCS.push(n);
      STORY.speakers[id] = d.name;
    });
  }
  function npcPlay(n, name, timeScale) {
    var next = n.actions[name];
    if (n.current === name) { next.timeScale = timeScale; return; }
    var prev = n.actions[n.current];
    next.reset(); next.timeScale = timeScale; next.setEffectiveWeight(1); next.play();
    if (prev) prev.crossFadeTo(next, 0.25, false);
    n.current = name;
  }
  function updateNpcs(dt) {
    if (zone !== 'village') return;
    for (var i = 0; i < NPCS.length; i++) {
      var n = NPCS[i], d = n.def, size = d.scale || 1;
      var toHero = Math.hypot(hero.pos.x - n.pos.x, hero.pos.z - n.pos.z);
      var speed = 0, want = n.heading;
      if (toHero < 3.2 || n.talking) {
        want = Math.atan2(hero.pos.x - n.pos.x, hero.pos.z - n.pos.z);       // turn to face Robin
      } else if (d.mode === 'patrol') {
        if (n.pause > 0) n.pause -= dt;
        else {
          var tgt = d.path[n.leg], dx = tgt[0] - n.pos.x, dz = tgt[1] - n.pos.z, dist = Math.hypot(dx, dz);
          if (dist < 0.15) { n.leg = 1 - n.leg; n.pause = 2.5; }
          else { speed = 1.05; want = Math.atan2(dx, dz); n.pos.x += dx / dist * speed * dt; n.pos.z += dz / dist * speed * dt; }
        }
      } else if (d.mode === 'circle') {
        var c = d.circle;
        n.ang += dt * 2.8 / c[2];
        var tx = c[0] + Math.sin(n.ang) * c[2], tz = c[1] + Math.cos(n.ang) * c[2];
        want = Math.atan2(tx - n.pos.x, tz - n.pos.z);
        speed = 2.8; n.pos.x = tx; n.pos.z = tz;
      } else if (d.face) {
        want = Math.atan2(d.face[0] - n.pos.x, d.face[1] - n.pos.z);
      }
      var diff = Math.atan2(Math.sin(want - n.heading), Math.cos(want - n.heading));
      n.heading += diff * Math.min(1, (speed > 2 ? 10 : 5) * dt);
      n.pos.y = VILLAGE.groundAt(n.pos.x, n.pos.z);
      if (speed > 2) npcPlay(n, 'Run', speed / (RUN_CLIP_SPEED * size));
      else if (speed > 0.1) npcPlay(n, 'Walk', speed / (WALK_CLIP_SPEED * size));
      else npcPlay(n, 'Idle', 1);
      n.mixer.update(dt);
      if (d.head) n.bones.Head.scale.setScalar(d.head);
      n.root.position.copy(n.pos);
      n.root.rotation.y = n.heading;
    }
  }
  function loadNpcPack() {
    fetchGz('npcs.b64.txt').then(parseBytes).then(function (g) { NPC_PACK = g; buildNpcs(); })
      .catch(function (err) { console.error('villagers', err); });
  }

  function loadVillage() {
    if (!window.fflate) return;
    function get(u) {
      return fetch(u).then(function (r) { if (!r.ok) throw new Error(u + ' ' + r.status); return r.text(); }).then(function (t) {
        var bin = atob(t.trim()), out = new Uint8Array(bin.length);
        for (var i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
        return out.buffer;
      });
    }
    // the enlarged village (new buildings, the town wall) loads alongside; the old village still builds without it
    var extP = Promise.all([get('village_ext.b64.txt'), fetch('village_ext.json').then(function (r) { return r.json(); })])
      .catch(function (err) { console.error('village ext', err); return null; });
    Promise.all([get('village.b64.txt'), get('village_inst.b64.txt'), fetch('village.json').then(function (r) { return r.json(); }), extP]).then(function (res) {
      var glb = fflate.gunzipSync(new Uint8Array(res[0]));
      var instBytes = fflate.gunzipSync(new Uint8Array(res[1]));
      var inst = new Float32Array(instBytes.buffer, instBytes.byteOffset, instBytes.byteLength / 4);
      var ext = res[3];
      var extG = ext ? parseBytes(fflate.gunzipSync(new Uint8Array(ext[0]))).catch(function (err) { console.error('village ext', err); return null; }) : Promise.resolve(null);
      Promise.all([parseBytes(glb), extG]).then(function (g) { buildVillage(g[0], inst, res[2], g[1], ext && ext[1]); }, function (err) { console.error('village', err); });
    }).catch(function (err) { console.error('village', err); });
  }
  function summon() {
    hero.locked = true; hero.cutscene = true; hero.vel.set(0, 0, 0);
    stage.classList.add('talking');
    var fl = $('flash');
    fl.style.transition = 'opacity 1.4s ease-in';
    fl.classList.add('on');
    setObjective();
    setTimeout(function () {
      if (!(VILLAGE && VILLAGE.ready)) $('flash-text').textContent = '…';
      whenVillage(enterVillage);
    }, 1500);
  }
  function enterVillage(options) {
    var quiet = options && options.quiet;     // the developer start skips the arrival lines
    var d = VILLAGE.data;
    zone = 'village';
    forestObjs.forEach(function (o) { o.visible = false; });
    VILLAGE.group.visible = true;
    BOXES = VILLAGE.boxes;
    POINTS = d.interact;
    doorFx.shake = 0;
    scene.background = srgb(0xe9d6bd);
    scene.fog.color = srgb(0xe2cfb4); scene.fog.near = 45; scene.fog.far = 520;
    sky.material.uniforms.top.value = new THREE.Color(0x4f86c8); sky.material.uniforms.low.value = new THREE.Color(0xf2d9b8);
    hemi.color = srgb(0xe4ecff); hemi.groundColor = srgb(0x7a7058); hemi.intensity = 0.72;
    sun.color = srgb(0xffd6a2); sun.intensity = 1.55;
    SUN_OFFSET.set(26, 15, -10);
    sun.shadow.camera.left = -24; sun.shadow.camera.right = 24; sun.shadow.camera.top = 24; sun.shadow.camera.bottom = -24;
    sun.shadow.camera.far = 90; sun.shadow.camera.updateProjectionMatrix();
    if (renderer.capabilities.maxTextureSize >= 4096) { sun.shadow.mapSize.set(4096, 4096); if (sun.shadow.map) { sun.shadow.map.dispose(); sun.shadow.map = null; } }
    hero.pos.set(d.spawn.x, VILLAGE.groundAt(d.spawn.x, d.spawn.z), d.spawn.z);
    hero.vel.set(0, 0, 0); hero.heading = d.spawn.heading; hero.onFloor = true;
    cam.yaw = 0; cam.pitch = 0.28; cam.goal = cam.distance = 6.2;
    focus.set(hero.pos.x, hero.pos.y + 1.3, hero.pos.z);
    hero.cutscene = false;
    play('Idle', 0);
    enterLand();
    var fl = $('flash');
    $('flash-text').textContent = '';
    setTimeout(function () { fl.style.transition = 'opacity 2.8s ease-out'; fl.classList.remove('on'); }, 250);
    if (quiet) { setObjective(); stage.classList.remove('talking'); hero.locked = false; return; }
    setTimeout(function () {
      say(STORY.village.arrive, function () { setObjective(); stage.classList.remove('talking'); });
    }, 1900);
  }

  // ---------------------------------------------------------------- the land around the village
  // Photoscanned (Poly Haven, CC0) ground, rocks, plants and trees on a 512 m height field that
  // continues the village's own terrain, under the HDRI sky it is lit by.
  var LAND = { ready: false, chunks: [], grid: {}, cell: 8, grass: null, sky: null };
  function decodeGrid(G) {
    var bin = atob(G.int16_le_base64), dv = new DataView(new ArrayBuffer(bin.length));
    for (var i = 0; i < bin.length; i++) dv.setUint8(i, bin.charCodeAt(i));
    var out = new Float32Array(bin.length / 2);
    for (var k = 0; k < out.length; k++) out[k] = dv.getInt16(k * 2, true) * G.unit;
    return out;
  }
  function gridSampler(G, hv) {
    return function (x, z) {
      var fx = Math.max(0, Math.min(G.size - 1.001, (x + G.half) / G.step));
      var fz = Math.max(0, Math.min(G.size - 1.001, (z + G.half) / G.step));
      var ix = Math.floor(fx), iz = Math.floor(fz), tx = fx - ix, tz = fz - iz, row = iz * G.size + ix;
      var a = hv[row] * (1 - tx) + hv[row + 1] * tx, b = hv[row + G.size] * (1 - tx) + hv[row + G.size + 1] * tx;
      return a * (1 - tz) + b * tz;
    };
  }
  function gridGeometry(G, hv, keepCell) {
    var n = G.size, pos = new Float32Array(n * n * 3), nor = new Float32Array(n * n * 3), idx = [];
    for (var iz = 0; iz < n; iz++) {
      for (var ix = 0; ix < n; ix++) {
        var k = iz * n + ix, x = -G.half + ix * G.step, z = -G.half + iz * G.step;
        pos[k * 3] = x; pos[k * 3 + 1] = hv[k]; pos[k * 3 + 2] = z;
        var hl = hv[iz * n + Math.max(ix - 1, 0)], hr = hv[iz * n + Math.min(ix + 1, n - 1)];
        var hu = hv[Math.max(iz - 1, 0) * n + ix], hd = hv[Math.min(iz + 1, n - 1) * n + ix];
        var nx = (hl - hr) / (2 * G.step), nz = (hu - hd) / (2 * G.step), L = Math.sqrt(nx * nx + 1 + nz * nz);
        nor[k * 3] = nx / L; nor[k * 3 + 1] = 1 / L; nor[k * 3 + 2] = nz / L;
      }
    }
    for (iz = 0; iz < n - 1; iz++) {
      for (ix = 0; ix < n - 1; ix++) {
        var x0 = -G.half + ix * G.step, z0 = -G.half + iz * G.step;
        if (!keepCell(x0, z0, x0 + G.step, z0 + G.step)) continue;
        var a = iz * n + ix, b = a + 1, c = a + n, d = c + 1;
        idx.push(a, c, b, b, c, d);
      }
    }
    var g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
    g.setIndex(new THREE.BufferAttribute(new Uint32Array(idx), 1));
    g.computeBoundingSphere();
    return g;
  }
  function groundTex(url, srgbColour) {
    var t = DETAIL.loader.load(url);
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.anisotropy = renderer.capabilities.getMaxAnisotropy();
    return t;
  }
  var GROUND_NAMES = ['meadow', 'forest', 'road', 'rock', 'sand'];
  function groundMaterial(splat, meta, water, procedural) {
    var m = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1, metalness: 0 });
    var uni = { uSplat: { value: splat }, uWater: { value: water }, uTiles: { value: [] }, uProc: { value: procedural ? 1 : 0 } };
    GROUND_NAMES.forEach(function (n, i) {
      uni['uC' + i] = { value: groundTex('ground/' + n + '_c.webp') };
      uni['uN' + i] = { value: groundTex('ground/' + n + '_n.webp') };
      uni.uTiles.value.push(1 / meta[n].tile);
    });
    m.onBeforeCompile = function (sh) {
      Object.keys(uni).forEach(function (k) { sh.uniforms[k] = uni[k]; });
      sh.vertexShader = 'varying vec3 vLW;\n' + sh.vertexShader.replace('#include <project_vertex>',
        '#include <project_vertex>\nvLW = (modelMatrix * vec4(transformed, 1.0)).xyz;');
      var decl = ['varying vec3 vLW;', 'uniform sampler2D uSplat; uniform float uWater; uniform float uTiles[5]; uniform float uProc;'];
      for (var i = 0; i < 5; i++) decl.push('uniform sampler2D uC' + i + '; uniform sampler2D uN' + i + ';');
      decl.push(
        'float gHash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }',
        'float gNoise(vec2 p) { vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);',
        '  return mix(mix(gHash(i), gHash(i + vec2(1, 0)), f.x), mix(gHash(i + vec2(0, 1)), gHash(i + vec2(1, 1)), f.x), f.y); }',
        'const mat2 gRot = mat2(0.4536, 0.8912, -0.8912, 0.4536);',
        'const mat2 gRotT = mat2(0.4536, -0.8912, 0.8912, 0.4536);',
        // two samplings of each layer, rotated and rescaled, blended by large-scale noise: no visible tiling
        // explicit gradients, so layers absent at this pixel can be skipped inside a branch without mip artefacts
        'void gLayer(sampler2D tc, sampler2D tn, vec2 uv, vec2 dx, vec2 dy, float k, out vec4 c, out vec4 n) {',
        '  vec2 uv2 = gRot * uv * 0.61 + 0.37; vec2 dx2 = gRot * dx * 0.61, dy2 = gRot * dy * 0.61;',
        '  vec4 c1 = textureGrad(tc, uv, dx, dy), c2 = textureGrad(tc, uv2, dx2, dy2);',
        '  vec4 n1 = textureGrad(tn, uv, dx, dy), n2 = textureGrad(tn, uv2, dx2, dy2);',
        '  n1.xy = n1.xy * 2.0 - 1.0; n2.xy = n2.xy * 2.0 - 1.0; n2.xy = gRotT * n2.xy;',
        '  c = mix(c1, c2, k); n = mix(n1, n2, k); }',
        'vec3 gAlbedo; float gRough; vec3 gTN;');
      sh.fragmentShader = decl.join('\n') + '\n' + sh.fragmentShader
        .replace('#include <map_fragment>', [
          'vec2 gW = vLW.xz;',
          'vec4 gS = texture2D(uSplat, vec2((gW.x + 256.0) / 512.0, 1.0 - (gW.y + 256.0) / 512.0));',
          'gS.a = clamp(1.0 - gS.r - gS.g - gS.b, 0.0, 1.0);',
          // beyond the mapped square (the horizon hills): turf and woodland floor, rock only where it is steep
          'if (uProc > 0.5 || max(abs(gW.x), abs(gW.y)) > 254.0) {',
          '  vec3 gFn = normalize(cross(dFdx(vLW), dFdy(vLW)));',
          '  float gR = smoothstep(0.82, 0.62, abs(gFn.y));',
          '  float gF = smoothstep(0.35, 0.65, gNoise(gW * 0.012));',
          '  gS = vec4((1.0 - gR) * (1.0 - gF), (1.0 - gR) * gF, 0.0, gR); }',
          'float gSand = smoothstep(uWater + 2.6, uWater + 1.5, vLW.y) * (1.0 - gS.a);',
          'float gW5[5]; gW5[0] = gS.r * (1.0 - gSand); gW5[1] = gS.g * (1.0 - gSand); gW5[2] = gS.b * (1.0 - gSand); gW5[3] = gS.a; gW5[4] = gSand;',
          'float gK = smoothstep(0.3, 0.7, gNoise(gW * 0.045) * 0.7 + gNoise(gW * 0.17) * 0.3);',
          'vec2 gUV = vec2(gW.x, -gW.y);',
          'vec4 gc[5]; vec4 gn[5]; vec2 gdx = dFdx(gUV), gdy = dFdy(gUV);',
          'for (int i = 0; i < 5; i++) { gc[i] = vec4(0.5); gn[i] = vec4(0.0, 0.0, 1.0, 0.9); }',
          'if (gW5[0] > 0.004) gLayer(uC0, uN0, gUV * uTiles[0], gdx * uTiles[0], gdy * uTiles[0], gK, gc[0], gn[0]);',
          'if (gW5[1] > 0.004) gLayer(uC1, uN1, gUV * uTiles[1], gdx * uTiles[1], gdy * uTiles[1], gK, gc[1], gn[1]);',
          'if (gW5[2] > 0.004) gLayer(uC2, uN2, gUV * uTiles[2], gdx * uTiles[2], gdy * uTiles[2], gK, gc[2], gn[2]);',
          'if (gW5[3] > 0.004) gLayer(uC3, uN3, gUV * uTiles[3], gdx * uTiles[3], gdy * uTiles[3], gK, gc[3], gn[3]);',
          'if (gW5[4] > 0.004) gLayer(uC4, uN4, gUV * uTiles[4], gdx * uTiles[4], gdy * uTiles[4], gK, gc[4], gn[4]);',
          // height-blended transitions: where layers meet, the higher detail (stones, tufts) wins
          'float gTop = 0.0; for (int i = 0; i < 5; i++) gTop = max(gTop, gc[i].a + gW5[i] * 1.6);',
          'float gSum = 0.0; float gB[5];',
          'for (int i = 0; i < 5; i++) { gB[i] = max(gc[i].a + gW5[i] * 1.6 - gTop + 0.35, 0.0) * step(0.004, gW5[i]); gSum += gB[i]; }',
          'gAlbedo = vec3(0.0); gRough = 0.0; vec2 gN2 = vec2(0.0);',
          'for (int i = 0; i < 5; i++) { float w = gB[i] / max(gSum, 1e-4); gAlbedo += pow(gc[i].rgb, vec3(2.2)) * w; gRough += gn[i].a * w; gN2 += gn[i].xy * w; }',
          'float gVar = 0.86 + 0.28 * gNoise(gW * 0.021);',
          'gAlbedo *= gVar;',
          'gAlbedo *= 1.0 - 0.55 * gW5[1];',          // woodland floor lies in the canopy's shade
          'gRough = mix(gRough, 1.0, 0.3);',
          'float gFar = smoothstep(40.0, 260.0, distance(cameraPosition, vLW));',
          'gN2 *= 1.0 - gFar * 0.75;',
          'gTN = normalize(vec3(gN2, 1.0));',
          'diffuseColor.rgb *= gAlbedo;'].join('\n'))
        .replace('#include <roughnessmap_fragment>', 'float roughnessFactor = gRough;')
        .replace('#include <normal_fragment_maps>', [
          '#include <normal_fragment_maps>',
          '{ vec3 tv = normalize((viewMatrix * vec4(1.0, 0.0, 0.0, 0.0)).xyz);',
          '  tv = normalize(tv - normal * dot(normal, tv)); vec3 bv = cross(normal, tv);',
          '  normal = normalize(tv * gTN.x + bv * gTN.y + normal * gTN.z); }'].join('\n'));
    };
    m.customProgramCacheKey = function () { return 'land-ground' + (procedural ? '-proc' : ''); };
    return m;
  }
  // Foliage cards carry normals bent out of the crown (or straight up for grass): their back faces must
  // not flip them downward, or half of every tree and tuft goes dark.
  function keepNormals(material) {
    var prev = material.onBeforeCompile;
    material.onBeforeCompile = function (sh, r) {
      if (prev) prev.call(this, sh, r);
      sh.fragmentShader = sh.fragmentShader.replace('#include <normal_fragment_begin>',
        THREE.ShaderChunk.normal_fragment_begin.replace('normal = normal * faceDirection;', ''));
    };
    var key = material.customProgramCacheKey ? material.customProgramCacheKey.bind(material) : function () { return ''; };
    material.customProgramCacheKey = function () { return key() + '|keepN'; };
  }
  // Instances grouped into 64 m chunks; each chunk shows its meshes only within a distance band.
  var LAND_CHUNK = 64;
  function landInstances(name, list, protoRoot, minD, maxD, opts) {
    opts = opts || {};
    var target = opts.group || LAND.group, chunks = opts.chunks || LAND.chunks;
    var meshes = [];
    protoRoot.updateMatrixWorld(true);
    protoRoot.traverse(function (o) { if (o.isMesh) meshes.push(o); });
    if (!meshes.length) return;
    var byChunk = {};
    list.forEach(function (it) {
      var key = Math.floor(it[0] / LAND_CHUNK) + ',' + Math.floor(it[2] / LAND_CHUNK);
      (byChunk[key] = byChunk[key] || []).push(it);
    });
    var mtx = new THREE.Matrix4(), q = new THREE.Quaternion(), s = new THREE.Vector3(), p = new THREE.Vector3(), up = new THREE.Vector3(0, 1, 0);
    var rootInv = new THREE.Matrix4().copy(protoRoot.matrixWorld).invert();
    Object.keys(byChunk).forEach(function (key) {
      var items = byChunk[key], cx = 0, cz = 0;
      items.forEach(function (it) { cx += it[0]; cz += it[2]; });
      var chunk = { x: cx / items.length, z: cz / items.length, min: minD, max: maxD, meshes: [] };
      meshes.forEach(function (src) {
        var local = new THREE.Matrix4().multiplyMatrices(rootInv, src.matrixWorld);
        // a geometry view per chunk sharing the buffers, with bounds that cover the whole chunk: the camera
        // and the sun's shadow pass then skip chunks they cannot see
        if (!src.geometry.boundingSphere) src.geometry.computeBoundingSphere();
        var geo = new THREE.BufferGeometry();
        for (var an in src.geometry.attributes) geo.setAttribute(an, src.geometry.attributes[an]);
        geo.setIndex(src.geometry.getIndex());
        var im = new THREE.InstancedMesh(geo, src.material, items.length);
        var cy = 0, reach = 0, maxS = 0;
        items.forEach(function (it, i) {
          q.setFromAxisAngle(up, it[3]); s.setScalar(it[4]); p.set(it[0], it[1] - (opts.sink || 0) * it[4], it[2]);
          mtx.compose(p, q, s).multiply(local);
          im.setMatrixAt(i, mtx);
          (it.refs = it.refs || []).push([im, i]);         // so one tree or rock can later be taken out of its chunk
          cy += it[1]; maxS = Math.max(maxS, it[4]);
        });
        cy /= items.length;
        items.forEach(function (it) { reach = Math.max(reach, Math.hypot(it[0] - chunk.x, it[1] - cy, it[2] - chunk.z)); });
        var bs = src.geometry.boundingSphere, off = new THREE.Vector3().copy(bs.center).applyMatrix4(local);
        geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(chunk.x, cy + off.y * maxS, chunk.z), reach + (bs.radius + off.length()) * maxS);
        im.instanceMatrix.needsUpdate = true;
        im.castShadow = opts.shadow !== false; im.receiveShadow = true;
        im.frustumCulled = true;
        im.visible = false;
        im.name = 'land_' + name;
        target.add(im);
        chunk.meshes.push(im);
      });
      chunks.push(chunk);
    });
  }
  function landCollider(x, z, r, grid) {
    var k = Math.floor(x / LAND.cell) + ',' + Math.floor(z / LAND.cell);
    grid = grid || LAND.grid;
    var c = { x: x, z: z, r: r };
    (grid[k] = grid[k] || []).push(c);
    return c;
  }
  function landPush(p, v) {
    var grid = zone === 'forest' ? FL.grid : LAND.grid;
    if (zone === 'forest' ? !FL.ready : !LAND.ready) return;
    var gx = Math.floor(p.x / LAND.cell), gz = Math.floor(p.z / LAND.cell);
    for (var dx = -1; dx <= 1; dx++) for (var dz = -1; dz <= 1; dz++) {
      var list = grid[(gx + dx) + ',' + (gz + dz)];
      if (!list) continue;
      for (var i = 0; i < list.length; i++) {
        var c = list[i], ex = p.x - c.x, ez = p.z - c.z, d = Math.hypot(ex, ez), reach = c.r + RADIUS;
        if (d < reach && d > 1e-4) { p.x = c.x + ex / d * reach; p.z = c.z + ez / d * reach; v.x *= 0.7; v.z *= 0.7; }
      }
    }
  }
  function grassField(tex, splatData, water, into) {
    // Thousands of crossed photo-tufts around the player, re-seeded on a world grid as Robin moves.
    var quads = [], uvs = [], nors = [], idx = [];
    for (var k = 0; k < 3; k++) {
      var a = k * Math.PI / 3, cx = Math.cos(a) * 0.5, cz = Math.sin(a) * 0.5, b = quads.length / 3;
      quads.push(-cx, 0, -cz, cx, 0, cz, cx, 1, cz, -cx, 1, -cz);
      uvs.push(0, 0, 1, 0, 1, 1, 0, 1);
      for (var j = 0; j < 4; j++) nors.push(0, 1, 0);
      idx.push(b, b + 1, b + 2, b, b + 2, b + 3);
    }
    var cells = [];
    for (var c = 0; c < 4; c++) {
      var g = new THREE.BufferGeometry();
      var u = uvs.map(function (val, i) { return (i % 2 ? (Math.floor(c / 2) + val) : (c % 2 + val)) * 0.5; });
      g.setAttribute('position', new THREE.Float32BufferAttribute(quads, 3));
      g.setAttribute('uv', new THREE.Float32BufferAttribute(u, 2));
      g.setAttribute('normal', new THREE.Float32BufferAttribute(nors, 3));
      g.setIndex(idx);
      var mat = new THREE.MeshStandardMaterial({ map: tex.c, alphaTest: 0.42, side: THREE.DoubleSide, roughness: 0.92, metalness: 0, envMapIntensity: 0.5 });
      addWind(mat, 0.12, 0.0);
      keepNormals(mat);
      var im = new THREE.InstancedMesh(g, mat, 7000);
      im.frustumCulled = false; im.receiveShadow = true; im.castShadow = false;
      im.count = 0;
      (into || LAND.group).add(im);
      cells.push(im);
    }
    var G = { cells: cells, at: new THREE.Vector2(1e9, 1e9), splat: splatData, water: water };
    return G;
  }
  function splatAt(G, x, z) {
    var ix = Math.max(0, Math.min(511, Math.floor(x + 256))), iz = Math.max(0, Math.min(511, Math.floor(z + 256)));
    var k = (iz * 512 + ix) * 4, d = G.splat, r = d[k] / 255, g = d[k + 1] / 255, b = d[k + 2] / 255;
    return [r, g, b, Math.max(0, 1 - r - g - b)];
  }
  function reseedGrass(G, px, pz) {
    var R = 40, step = 0.5, counts = [0, 0, 0, 0], mtx = new THREE.Matrix4(), q = new THREE.Quaternion(), s = new THREE.Vector3(), p = new THREE.Vector3(), up = new THREE.Vector3(0, 1, 0);
    var x0 = Math.floor((px - R) / step), x1 = Math.ceil((px + R) / step), z0 = Math.floor((pz - R) / step), z1 = Math.ceil((pz + R) / step);
    for (var gx = x0; gx <= x1; gx++) {
      for (var gz = z0; gz <= z1; gz++) {
        var h1 = Math.sin(gx * 127.1 + gz * 311.7) * 43758.5453; h1 -= Math.floor(h1);
        var h2 = Math.sin(gx * 269.5 + gz * 183.3) * 43758.5453; h2 -= Math.floor(h2);
        var x = (gx + h1) * step, z = (gz + h2) * step;
        var dd = (x - px) * (x - px) + (z - pz) * (z - pz);
        if (dd > R * R) continue;
        if (!G.weight && Math.hypot(x, z - 3) < 23) continue;
        if (!G.weight && roomFloorAt(x, z, 0.7)) continue;        // no grass through the floor of a room
        var meadow, y;
        if (G.weight) { meadow = G.weight(x, z); y = G.ground(x, z); }
        else { var w = splatAt(G, x, z); meadow = w[0] + w[1] * 0.35; y = LAND.groundAt(x, z); }
        if (meadow <= 0 || y < G.water + 0.9) continue;
        if (h1 * 0.7 + h2 * 0.3 > meadow * (1.05 - dd / (R * R) * 0.45)) continue;
        var cell = Math.floor(h2 * 3.999), im = G.cells[cell], n = counts[cell];
        if (n >= im.instanceMatrix.count) continue;
        q.setFromAxisAngle(up, h1 * 6.283); var sc = 0.5 + h2 * 0.45; s.set(sc * 1.1, sc * (0.8 + h1 * 0.5), sc * 1.1);
        p.set(x, y - 0.03, z);
        mtx.compose(p, q, s); im.setMatrixAt(n, mtx); counts[cell] = n + 1;
      }
    }
    G.cells.forEach(function (im, i) { im.count = counts[i]; im.instanceMatrix.needsUpdate = true; });
    G.at.set(px, pz);
  }
  function updateLand() {
    if (zone === 'forest' && FL.ready) {
      var fx = camera.position.x, fz = camera.position.z;
      for (var fi = 0; fi < FL.chunks.length; fi++) {
        var fc = FL.chunks[fi], fd = Math.hypot(fc.x - fx, fc.z - fz), fon = fd >= fc.min && fd < fc.max;
        for (var fk = 0; fk < fc.meshes.length; fk++) fc.meshes[fk].visible = fon;
      }
      if (FL.grass.at.distanceTo(new THREE.Vector2(hero.pos.x, hero.pos.z)) > 6) reseedGrass(FL.grass, hero.pos.x, hero.pos.z);
      return;
    }
    if (!LAND.ready || zone !== 'village') return;
    var cx = camera.position.x, cz = camera.position.z;
    for (var i = 0; i < LAND.chunks.length; i++) {
      var ch = LAND.chunks[i], d = Math.hypot(ch.x - cx, ch.z - cz), on = d >= ch.min && d < ch.max;
      for (var k = 0; k < ch.meshes.length; k++) ch.meshes[k].visible = on;
    }
    if (LAND.grass && LAND.grass.at.distanceTo(new THREE.Vector2(hero.pos.x, hero.pos.z)) > 6) reseedGrass(LAND.grass, hero.pos.x, hero.pos.z);
  }
  function landLights() {
    // the HDRI's sun: direction, warm colour; the sky itself fills the shade
    var s = LAND.sky;
    if (!s) return;
    SUN_OFFSET.set(s.sun[0], s.sun[1], s.sun[2]).multiplyScalar(40);
    sun.color = srgb(0xfff1df); sun.intensity = 2.3;
    hemi.intensity = 0.0;
    renderer.toneMappingExposure = 0.95;
    scene.fog.color = srgb(0xc9d5df); scene.fog.near = 80; scene.fog.far = 900;
    scene.background = srgb(0xc9d5df);
    if (LAND.skyMesh) { sky.visible = false; LAND.skyMesh.visible = true; }
    sun.shadow.camera.left = -36; sun.shadow.camera.right = 36; sun.shadow.camera.top = 36; sun.shadow.camera.bottom = -36;
    sun.shadow.camera.far = 140; sun.shadow.camera.updateProjectionMatrix();
  }
  function buildLand(data, trees, props, propMeta, skyInfo, splatData) {
    LAND.group = new THREE.Group();
    LAND.group.name = 'land';
    LAND.group.visible = false;
    var near = decodeGrid(data.near), far = decodeGrid(data.far);
    LAND.groundAt = gridSampler(data.near, near);
    LAND.farAt = gridSampler(data.far, far);
    LAND.water = data.water;
    LAND.sky = skyInfo;
    var splat = DETAIL.loader.load('land_splat.png');
    splat.minFilter = THREE.LinearFilter; splat.wrapS = splat.wrapT = THREE.ClampToEdgeWrapping; splat.generateMipmaps = false;
    var meta = window.MEGALITH_GROUND || {};
    var gm = groundMaterial(splat, meta, data.water);
    var nearGeo = gridGeometry(data.near, near, function (x0, z0, x1, z1) {
      return Math.hypot((x0 + x1) / 2, (z0 + z1) / 2 - 3) > 20;      // under the old ground's dissolving edge too
    });
    var nearMesh = new THREE.Mesh(nearGeo, gm);
    nearMesh.receiveShadow = true; nearMesh.name = 'land_near';
    gm.polygonOffset = true; gm.polygonOffsetFactor = 1; gm.polygonOffsetUnits = 2;
    LAND.group.add(nearMesh);
    var fm = gm.clone(); fm.onBeforeCompile = gm.onBeforeCompile; fm.customProgramCacheKey = gm.customProgramCacheKey;
    fm.polygonOffsetFactor = 4; fm.polygonOffsetUnits = 8;
    var farGeo = gridGeometry(data.far, far, function (x0, z0, x1, z1) { return Math.max(Math.abs(x0), Math.abs(x1), Math.abs(z0), Math.abs(z1)) > 250; });
    var farMesh = new THREE.Mesh(farGeo, fm);
    farMesh.name = 'land_far';
    LAND.group.add(farMesh);
    // the open sea out to the horizon
    var seaMat = new THREE.MeshStandardMaterial({ color: srgb(0x163039), roughness: 0.14, metalness: 0.0 });
    seaMat.onBeforeCompile = function (sh) {          // rolling swell: layered moving waves perturb the normal
      sh.uniforms.uTime = FOREST.wind;
      sh.vertexShader = 'varying vec3 vSW;\n' + sh.vertexShader.replace('#include <project_vertex>', '#include <project_vertex>\nvSW = (modelMatrix * vec4(transformed, 1.0)).xyz;');
      sh.fragmentShader = 'uniform float uTime; varying vec3 vSW;\n' + sh.fragmentShader.replace('#include <normal_fragment_maps>', [
        '#include <normal_fragment_maps>',
        '{ vec2 p = vSW.xz; float t = uTime;',
        '  vec2 g = vec2(0.0);',
        '  g += vec2(0.8, 0.6) * cos(dot(p, vec2(0.8, 0.6)) * 0.35 + t * 0.9) * 0.10;',
        '  g += vec2(-0.5, 0.86) * cos(dot(p, vec2(-0.5, 0.86)) * 0.9 + t * 1.6) * 0.06;',
        '  g += vec2(0.97, -0.24) * cos(dot(p, vec2(0.97, -0.24)) * 2.1 + t * 2.4) * 0.035;',
        '  g += vec2(-0.2, -0.98) * cos(dot(p, vec2(-0.2, -0.98)) * 4.7 + t * 3.3) * 0.02;',
        '  float fade = 1.0 - smoothstep(60.0, 600.0, distance(cameraPosition, vSW));',
        '  vec3 nW = normalize(vec3(-g.x * fade, 1.0, -g.y * fade));',
        '  normal = normalize((viewMatrix * vec4(nW, 0.0)).xyz); }'].join('\n'));
    };
    var sea = new THREE.Mesh(new THREE.CircleGeometry(2400, 96), seaMat);
    sea.rotation.x = -Math.PI / 2; sea.position.y = data.water - 0.05; sea.name = 'land_sea';
    LAND.group.add(sea);
    // trees: full near, the light far version beyond
    var protos = {};
    trees.scene.traverse(function (o) { if (/^tree_/.test(o.name) && o.parent === trees.scene) protos[o.name] = o; });
    props.scene.traverse(function (o) { if (o.isMesh && o.parent === props.scene) protos[o.name] = o; });
    Object.keys(data.scatter).forEach(function (name) {
      var list = data.scatter[name], pm = propMeta[name];
      if (/^tree_/.test(name)) {
        if (protos[name]) landInstances(name, list, protos[name], 0, 95, { sink: 0.15 });
        if (protos[name + '_far']) landInstances(name + '_far', list, protos[name + '_far'], 85, 420, { sink: 0.15, shadow: false });
        else if (protos[name]) landInstances(name + '_mid', list, protos[name], 95, 420, { sink: 0.15, shadow: false });
        list.forEach(function (it) {
          var r = (/fir/.test(name) ? 0.42 : /ash/.test(name) ? 0.35 : 0.55) * it[4];
          resAdd('tree', name, it, landCollider(it[0], it[2], r), r);
        });
        return;
      }
      if (!protos[name] || !pm) return;
      var kind = pm.kind, maxD = { rock: 320, cliff: 400, stump: 170, log: 170, roots: 150, twigs: 70, fern: 75, shrub: 110, weed: 60, flower: 55, moss: 40, pebble: 40 }[kind] || 90;
      landInstances(name, list, protos[name], 0, maxD, { shadow: kind !== 'flower' && kind !== 'pebble' && kind !== 'moss', sink: kind === 'rock' || kind === 'cliff' ? 0.06 : 0.0 });
      if (kind === 'rock' || kind === 'cliff' || kind === 'stump' || kind === 'log') {
        list.forEach(function (it) {
          var r = Math.min(pm.size[0], pm.size[2]) * 0.42 * it[4], col = landCollider(it[0], it[2], r);
          if (kind === 'rock' && Math.max(pm.size[0], pm.size[1], pm.size[2]) * it[4] < 3.2) resAdd('rock', name, it, col, r, pm.size[1] * it[4]);
        });
      }
    });
    LAND.protos = protos;
    // leaves and fronds: alpha-tested, double-sided, with wind, and shadows cut to their shapes
    LAND.group.traverse(function (o) {
      if (!o.isMesh || !o.material) return;
      var m = o.material;
      if (m.userData.landTuned) return;
      m.userData.landTuned = true;
      if (/^leaf_/.test(m.name)) {
        m.side = THREE.DoubleSide; m.alphaTest = 0.4; m.transparent = false;
        m.roughness = 0.9; m.envMapIntensity = 0.45;          // foliage scatters light; no glassy sky sheen
        addWind(m, /fir/.test(m.name) ? 0.0009 : 0.0016, 2.0);
        keepNormals(m);
        o.customDepthMaterial = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking, map: m.map, alphaTest: 0.4 });
      } else if (/(grass|fern|shrub|nettle|dandelion)/.test(m.name)) {
        m.side = THREE.DoubleSide;
        addWind(m, 0.05, 0.0);
      }
      if (m.map) m.map.anisotropy = renderer.capabilities.getMaxAnisotropy();
    });
    LAND.group.traverse(function (o) { if (o.isInstancedMesh && /^leaf_/.test(o.material.name)) o.customDepthMaterial = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking, map: o.material.map, alphaTest: 0.4 }); });
    // grass tufts near the player
    var gt = { c: groundTex('ground/grass_c.webp'), n: groundTex('ground/grass_n.webp') };
    gt.c.encoding = THREE.sRGBEncoding; gt.c.wrapS = gt.c.wrapT = THREE.ClampToEdgeWrapping;
    LAND.grass = grassField(gt, splatData, data.water);
    // sky dome: the HDRI's own tonemapped photograph, sampled by direction like the light probe
    var skyTex = DETAIL.loader.load('sky/sky_bg.jpg');
    skyTex.minFilter = THREE.LinearFilter; skyTex.generateMipmaps = false;
    LAND.skyMesh = new THREE.Mesh(new THREE.SphereGeometry(800, 48, 24), new THREE.ShaderMaterial({
      side: THREE.BackSide, depthWrite: false, fog: false, uniforms: { map: { value: skyTex } },
      vertexShader: 'varying vec3 d; void main(){ d = position; vec4 p = projectionMatrix * modelViewMatrix * vec4(position, 1.0); gl_Position = p.xyww; }',
      fragmentShader: 'uniform sampler2D map; varying vec3 d; void main(){ vec3 n = normalize(d); vec2 uv = vec2(atan(n.z, n.x) / 6.2831853 + 0.5, asin(clamp(n.y, -1.0, 1.0)) / 3.1415927 + 0.5); gl_FragColor = vec4(texture2D(map, uv).rgb, 1.0); }'
    }));
    LAND.skyMesh.renderOrder = -2; LAND.skyMesh.frustumCulled = false; LAND.skyMesh.visible = false;
    scene.add(LAND.skyMesh);
    scene.add(LAND.group);
    freeze(LAND.group);
    precompile(LAND.group);
    LAND.ready = true;
    LAND.protos = protos; LAND.gt = gt;
    buildForestLand(protos, gt);
    if (zone === 'village') enterLand();
  }

  // ---------------------------------------------------------------- the ruins, dressed like the land
  // Same scanned trees, plants, stones, grass tufts, ground layers and HDRI sky as around the village;
  // the standing stones themselves are untouched.
  var FL = { ready: false, chunks: [], grid: {}, grass: null, group: null };
  function buildForestLand(protos, gt) {
    if (!forestScene || !FOREST.scatter || FL.ready || !protos) return;
    FL.group = new THREE.Group();
    FL.group.name = 'forest_land';
    Object.keys(FOREST.scatter).forEach(function (name) {
      var list = FOREST.scatter[name], proto = protos[name], tree = /^tree_/.test(name);
      if (!proto) return;
      landInstances(name, list, proto, 0, tree ? 95 : 70, { group: FL.group, chunks: FL.chunks, sink: tree ? 0.15 : 0,
        shadow: !/dandelion|rock_0|stone_01/.test(name) });
      if (tree && protos[name + '_far']) landInstances(name + '_far', list, protos[name + '_far'], 85, 320, { group: FL.group, chunks: FL.chunks, sink: 0.15, shadow: false });
      if (tree) list.forEach(function (it) { landCollider(it[0], it[2], (/fir/.test(name) ? 0.42 : /ash/.test(name) ? 0.35 : 0.55) * it[4], FL.grid); });
      else if (/stump|root|rock_moss/.test(name)) list.forEach(function (it) { landCollider(it[0], it[2], 0.45 * it[4], FL.grid); });
    });
    var meta = window.MEGALITH_GROUND || {}, gm = null;
    forestScene.traverse(function (o) {
      if (!o.isMesh) return;
      if (/^terrain/.test(o.name)) {
        gm = gm || groundMaterial(null, meta, -100, true);
        o.material = gm; o.receiveShadow = true;
      }
      if (/^(trunks|leaves|moon)$/.test(o.name)) o.visible = false;
      if (o.name === 'mountains') {              // the far peaks in the same scanned rock and turf as the ground
        o.material = groundMaterial(null, meta, -100, true);
        o.receiveShadow = false; o.castShadow = false;
      }
    });
    var boxes = WORLD.colliders;
    FL.grass = grassField(gt, null, -100, FL.group);
    FL.grass.ground = function (x, z) { return groundAt(x, z); };
    FL.grass.weight = function (x, z) {           // no tufts growing through the standing stones
      for (var i = 0; i < boxes.length; i++) { var b = boxes[i]; if (Math.abs(x - b.x) < b.hx + 0.6 && Math.abs(z - b.z) < b.hz + 0.6) return 0; }
      return 0.85;
    };
    FL.group.traverse(function (o) {
      if (o.isInstancedMesh && /^leaf_/.test(o.material.name)) o.customDepthMaterial = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking, map: o.material.map, alphaTest: 0.4 });
    });
    forestScene.add(FL.group);
    forestObjs.push(FL.group);
    freeze(FL.group);
    FL.ready = true;
    if (zone === 'forest') forestLights();
    precompile(FL.group);
  }
  // Static things: compute their matrices once instead of every frame.
  function freeze(root) { root.updateMatrixWorld(true); root.traverse(function (o) { o.matrixAutoUpdate = false; }); }
  // Compile a group's shaders now (briefly visible to the renderer), so walking into it later never stutters.
  function precompile(root) {
    var was = [], p = root;
    while (p) { was.push([p, p.visible]); p.visible = true; p = p.parent; }
    try { renderer.compile(scene, camera); } catch (e) { /* ignore */ }
    was.forEach(function (w) { w[0].visible = w[1]; });
  }
  function forestLights() {
    if (!FL.ready) return;
    landLights();
    if (LAND.envTex) scene.environment = LAND.envTex;
  }
  function enterLand() {
    if (!LAND.ready) return;
    LAND.group.visible = true;
    if (VILLAGE) { VILLAGE.bounds = { x: 0, z: 0, r: 355 }; BOXES = VILLAGE.boxes.concat(LAND.boxes || []); }
    landLights();
    if (LAND.envTex) scene.environment = LAND.envTex;
  }

  // ---------------------------------------------------------------- the castle on its island
  function buildCastle(gltf, info) {
    var g = gltf.scene, o0 = info.origin;
    g.position.set(o0[0], o0[1], o0[2]);
    g.traverse(function (o) {
      if (!o.isMesh) return;
      o.castShadow = true; o.receiveShadow = true;
      var m = o.material;
      if (m.map) m.map.anisotropy = renderer.capabilities.getMaxAnisotropy();
      if (/glow/.test(m.name)) { m.emissiveIntensity = 2.5; o.castShadow = false; }
      if (/cloth|carpet/.test(m.name)) m.side = THREE.DoubleSide;
    });
    LAND.group.add(g);
    freeze(g);
    precompile(g);
    // the gate passage between the bridge's end and the courtyard floor needs a floor of its own
    var cols = info.colliders.concat([{ x: 0, z: 20.3, yaw: 0, hx: 2.3, hz: 2.8, y0: -1, y1: 0.05 },
      { x: 0, z: 2.0, yaw: 0, hx: 1.7, hz: 0.85, y0: 0, y1: 1.2 }]);      // ...and so does the keep's doorway, from the steps to the hall
    LAND.boxes = (LAND.boxes || []).concat(boxesOf(cols.map(function (c) {
      return { x: o0[0] + c.x, z: o0[2] + c.z, yaw: -c.yaw, hx: c.hx, hz: c.hz, y0: o0[1] + c.y0, y1: o0[1] + c.y1 };
    })));
    // warm light from the hall's fire and candles
    LAND.castle = { x: o0[0], y: o0[1], z: o0[2], keep: info.keep };
    var k = info.keep, hall = new THREE.PointLight(srgb(0xffa45a), 2.4, 20, 2);
    hall.position.set(o0[0] + k.x + 6.5, o0[1] + k.floor + 2.2, o0[2] + k.z);
    LAND.group.add(hall);
    var chand = new THREE.PointLight(srgb(0xffc27a), 2.0, 24, 2);          // the two candle rings
    chand.position.set(o0[0] + k.x, o0[1] + 9.5, o0[2] + k.z);
    LAND.group.add(chand);
    if (zone === 'village' && VILLAGE) BOXES = VILLAGE.boxes.concat(LAND.boxes);
  }

  // ---------------------------------------------------------------- monsters in the wild
  var MONSTERS = [];
  var MONSTER_SPOTS = {
    horror: [[-82, 44], [-118, 86], [-64, 132], [26, 150], [-170, 14], [-150, 150], [-40, 205]],
    golem: [[-128, -58], [-196, 118], [64, 196], [-205, -40]]
  };
  function spawnMonstersOld(kind, gltf) {
    MONSTER_SPOTS[kind].forEach(function (s, i) {
      var root = gltf.scene.clone(true), parts = {};
      root.traverse(function (o) {
        if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; }
        parts[o.name] = { node: o, rest: o.quaternion.clone(), pos: o.position.clone() };
      });
      var y = LAND.groundAt(s[0], s[1]);
      if (y < LAND.water + 0.3) return;
      root.position.set(s[0], y, s[1]);
      LAND.group.add(root);
      MONSTERS.push({ kind: kind, root: root, parts: parts, home: new THREE.Vector2(s[0], s[1]), heading: Math.random() * 6.28,
                      target: null, wait: Math.random() * 3, phase: Math.random() * 6.28, t: Math.random() * 10,
                      speed: kind === 'golem' ? 0.9 : 1.2, radius: kind === 'golem' ? 1.2 : 0.9 });
    });
  }
  var _q = new THREE.Quaternion(), _e = new THREE.Euler();
  function pose(m, name, rx, ry, rz) {
    var p = m.parts[name];
    if (!p) return;
    _e.set(rx || 0, ry || 0, rz || 0);
    p.node.quaternion.copy(p.rest).multiply(_q.setFromEuler(_e));
  }
  function updateMonstersOld(dt) {
    if (!LAND.ready || zone !== 'village' || !MONSTERS.length) return;
    for (var i = 0; i < MONSTERS.length; i++) {
      var m = MONSTERS[i], r = m.root;
      m.t += dt;
      var dxh = hero.pos.x - r.position.x, dzh = hero.pos.z - r.position.z, dh = Math.hypot(dxh, dzh);
      var moving = false;
      if (dh < 9) {                                            // it has noticed Robin: turn and watch
        var want = Math.atan2(dxh, dzh), diff = Math.atan2(Math.sin(want - m.heading), Math.cos(want - m.heading));
        m.heading += diff * Math.min(1, 2.5 * dt);
      } else if (m.wait > 0) {
        m.wait -= dt;
      } else {
        if (!m.target) {
          var a = Math.random() * 6.28, d = 4 + Math.random() * 12;
          m.target = new THREE.Vector2(m.home.x + Math.cos(a) * d, m.home.y + Math.sin(a) * d);
        }
        var tx = m.target.x - r.position.x, tz = m.target.y - r.position.z, td = Math.hypot(tx, tz);
        if (td < 0.6 || LAND.groundAt(m.target.x, m.target.y) < LAND.water + 0.4) { m.target = null; m.wait = 2 + Math.random() * 5; }
        else {
          var want2 = Math.atan2(tx, tz), diff2 = Math.atan2(Math.sin(want2 - m.heading), Math.cos(want2 - m.heading));
          m.heading += diff2 * Math.min(1, 2 * dt);
          if (Math.abs(diff2) < 0.6) {
            r.position.x += Math.sin(m.heading) * m.speed * dt; r.position.z += Math.cos(m.heading) * m.speed * dt;
            moving = true;
          }
        }
      }
      r.position.y = LAND.groundAt(r.position.x, r.position.z);
      r.rotation.y = m.heading;
      m.phase += dt * (moving ? (m.kind === 'golem' ? 3.2 : 5.0) : 1.2);
      var ph = m.phase, w = moving ? 1 : 0.15;
      if (m.kind === 'golem') {
        pose(m, 'thigh_l', Math.sin(ph) * 0.45 * w); pose(m, 'thigh_r', -Math.sin(ph) * 0.45 * w);
        pose(m, 'shin_l', Math.max(0, -Math.cos(ph)) * 0.5 * w); pose(m, 'shin_r', Math.max(0, Math.cos(ph)) * 0.5 * w);
        pose(m, 'arm_l', -Math.sin(ph) * 0.35 * w, 0, 0.05); pose(m, 'arm_r', Math.sin(ph) * 0.35 * w, 0, -0.05);
        pose(m, 'fore_l', -0.25 - Math.max(0, Math.sin(ph)) * 0.2 * w); pose(m, 'fore_r', -0.25 - Math.max(0, -Math.sin(ph)) * 0.2 * w);
        pose(m, 'torso', 0.04 + Math.sin(m.t * 1.3) * 0.025, Math.sin(ph) * 0.06 * w);
        pose(m, 'head', Math.sin(m.t * 0.7) * 0.05, dh < 9 ? 0 : Math.sin(m.t * 0.4) * 0.3);
        m.parts.pelvis && (m.parts.pelvis.node.position.y = m.parts.pelvis.pos.y + Math.abs(Math.sin(ph)) * 0.05 * w);
      } else {
        for (var t = 0; t < 6; t++) for (var k = 0; k < 5; k++) {
          var wave = Math.sin(m.t * (moving ? 4.5 : 1.6) - k * 0.7 + t * 1.05);
          pose(m, 'tent_' + t + '_' + k, wave * (0.1 + 0.05 * k) * (moving ? 1.6 : 1), 0, Math.cos(m.t * 1.1 + t + k * 0.5) * 0.08);
        }
        pose(m, 'arm_l', Math.sin(m.t * 1.7) * 0.15, 0, Math.sin(m.t * 0.9) * 0.1); pose(m, 'arm_r', Math.sin(m.t * 1.7 + 1) * 0.15, 0, -Math.sin(m.t * 0.9) * 0.1);
        pose(m, 'fin_l', 0, 0, Math.sin(m.t * 2.3) * 0.18); pose(m, 'fin_r', 0, 0, -Math.sin(m.t * 2.3) * 0.18);
        pose(m, 'head', Math.sin(m.t * 1.1) * 0.08, Math.sin(m.t * 0.6) * 0.2);
        m.parts.body && (m.parts.body.node.position.y = m.parts.body.pos.y + Math.sin(m.t * (moving ? 6 : 2)) * 0.04);
      }
      // solid: Robin cannot walk through them
      if (dh < m.radius + RADIUS && dh > 1e-4) { hero.pos.x = r.position.x + dxh / dh * (m.radius + RADIUS); hero.pos.z = r.position.z + dzh / dh * (m.radius + RADIUS); }
    }
  }
  // ---------------------------------------------------------------- monsters: roaming, hunting, fighting
  // Each monster roams the whole land (its home drifts from walk to walk), notices Robin within 16 m,
  // hunts him down, winds up and strikes. Robin's blows flash and stagger it, knock it back and can tear
  // off a limb, which falls under physics (gravity, bounce, tumble); a killed monster bursts apart.
  // Three kills open the bridge to the castle.
  var COMBAT = { strike: 0, kills: 0, need: 3, debris: [], sparks: [], shake: 0, gateMsgAt: -99, hurtFx: 0 };
  var KIND = {
    horror: { hp: 60, walk: 1.25, run: 3.6, reach: 1.9, dmg: 9, windup: 0.42, cool: 1.5, radius: 0.9, blood: 0x4a0c0c,
              limbs: ['hand_l', 'hand_r', 'fore_l', 'fore_r', 'fin_l', 'fin_r', 'tent_0_3', 'tent_1_2', 'tent_2_3', 'tent_3_2', 'tent_4_3', 'tent_5_2', 'tent_0_1', 'tent_3_1', 'head'] },
    golem: { hp: 120, walk: 0.95, run: 2.3, reach: 2.5, dmg: 15, windup: 0.6, cool: 2.0, radius: 1.25, blood: 0x8a8f96,
             limbs: ['hand_l', 'hand_r', 'fore_l', 'fore_r', 'arm_l', 'arm_r', 'head'] }
  };
  function landOk(x, z) {
    if (!LAND.ready || Math.abs(x) > 245 || Math.abs(z) > 245) return false;
    if (LAND.groundAt(x, z) < LAND.water + 0.5) return false;
    if (Math.hypot(x, z - 3) < 44 || inVillage(x, z, 10)) return false;              // not inside the village or by its wall
    if (LAND.castle && Math.hypot(x - LAND.castle.x, z - LAND.castle.z) < 75) return false;
    return true;
  }
  function spawnMonsters(kind, gltf) {
    var count = kind === 'horror' ? 16 : 9, seed = kind === 'horror' ? 7 : 19, made = 0, tries = 0;
    function rnd() { seed = (seed * 16807) % 2147483647; return seed / 2147483647; }
    while (made < count && tries++ < 600) {
      var x = rnd() * 460 - 230, z = rnd() * 460 - 230;
      if (!landOk(x, z)) continue;
      var root = gltf.scene.clone(true), parts = {};
      root.traverse(function (o) {
        if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; o.material = o.material.clone(); }   // own materials: hit flashes
        parts[o.name] = { node: o, rest: o.quaternion.clone(), pos: o.position.clone() };
      });
      root.position.set(x, LAND.groundAt(x, z), z);
      LAND.group.add(root);
      var K = KIND[kind];
      MONSTERS.push({ kind: kind, K: K, root: root, parts: parts, hp: K.hp, state: 'roam', home: new THREE.Vector2(x, z), heading: rnd() * 6.28,
                      target: null, wait: rnd() * 3, phase: rnd() * 6.28, t: rnd() * 10, timer: 0, cool: 0, flash: 0, knock: new THREE.Vector2(),
                      severed: {}, radius: K.radius });
      made++;
    }
  }
  var _hitV = new THREE.Vector3(), _wp = new THREE.Vector3(), _wq = new THREE.Quaternion(), _ws = new THREE.Vector3();
  function burst(pos, colour, n, speed) {
    var g = new THREE.BufferGeometry(), p = new Float32Array(n * 3), vel = [];
    for (var i = 0; i < n; i++) {
      p[i * 3] = pos.x; p[i * 3 + 1] = pos.y; p[i * 3 + 2] = pos.z;
      vel.push(new THREE.Vector3((Math.random() - 0.5) * speed, Math.random() * speed * 0.9 + 1, (Math.random() - 0.5) * speed));
    }
    g.setAttribute('position', new THREE.BufferAttribute(p, 3));
    var m = new THREE.PointsMaterial({ color: colour, size: 0.09, transparent: true, opacity: 1, depthWrite: false });
    var pts = new THREE.Points(g, m);
    pts.frustumCulled = false;
    scene.add(pts);
    COMBAT.sparks.push({ pts: pts, vel: vel, life: 1.1 });
  }
  function sever(m, name, impulse) {
    var p = m.parts[name];
    if (!p || m.severed[name] || !p.node.parent) return false;
    var node = p.node;
    node.updateWorldMatrix(true, true);
    node.matrixWorld.decompose(_wp, _wq, _ws);
    node.parent.remove(node);
    node.position.copy(_wp); node.quaternion.copy(_wq); node.scale.copy(_ws);
    LAND.group.add(node);
    node.traverse(function (o) { m.severed[o.name] = true; o.matrixAutoUpdate = true; });
    var dir = impulse.clone();
    COMBAT.debris.push({ obj: node, vel: new THREE.Vector3(dir.x * 3.5 + (Math.random() - 0.5) * 2, 3 + Math.random() * 3, dir.z * 3.5 + (Math.random() - 0.5) * 2),
                         spin: new THREE.Vector3((Math.random() - 0.5) * 10, (Math.random() - 0.5) * 10, (Math.random() - 0.5) * 10), life: 9, rest: false });
    burst(_wp, m.K.blood, m.kind === 'golem' ? 26 : 34, 4);
    return true;
  }
  function heroStrike() {
    var fx = Math.sin(hero.heading), fz = Math.cos(hero.heading), any = false;
    for (var i = 0; i < MONSTERS.length; i++) {
      var m = MONSTERS[i];
      if (m.state === 'dead') continue;
      var dx = m.root.position.x - hero.pos.x, dz = m.root.position.z - hero.pos.z, d = Math.hypot(dx, dz);
      var W = WEAPONS[INV.held] || WEAPONS[0];
      if (d > W.reach + m.radius || (dx * fx + dz * fz) / Math.max(d, 1e-3) < W.arc) continue;
      any = true;
      var dmg = W.dmg * (0.85 + Math.random() * 0.3) * buffMult();
      m.hp -= dmg;
      m.flash = 0.18; m.state = m.hp > 0 ? 'hurt' : 'dead'; m.timer = 0.35;
      m.knock.set(dx / d * 5.5, dz / d * 5.5);
      _hitV.set(dx / d, 0, dz / d);
      _wp.copy(m.root.position); _wp.y += m.kind === 'golem' ? 1.6 : 1.0;
      burst(_wp, m.K.blood, 16, 3);
      if (m.hp > 0) {
        if (Math.random() < W.sever * (m.kind === 'golem' ? 0.75 : 1)) {        // tear off a limb
          var pool = m.K.limbs.filter(function (n) { return n !== 'head' && !m.severed[n]; });
          if (pool.length) sever(m, pool[Math.floor(Math.random() * pool.length)], _hitV);
        }
      } else {
        // it bursts apart: every limb flies, the body slumps and sinks
        m.K.limbs.concat(m.kind === 'golem' ? ['torso', 'thigh_l', 'thigh_r'] : ['tent_0_0', 'tent_1_0', 'tent_2_0', 'tent_3_0', 'tent_4_0', 'tent_5_0']).forEach(function (n) {
          var a = Math.random() * 6.28;
          sever(m, n, new THREE.Vector3(Math.cos(a) * 0.8 + _hitV.x, 0, Math.sin(a) * 0.8 + _hitV.z));
        });
        m.timer = 0;
        dropLoot(m);
        COMBAT.kills++;
        setObjective();
        showArea(COMBAT.kills >= COMBAT.need ? '성으로 가는 다리가 열렸다' : '몬스터 처치  ' + COMBAT.kills + ' / ' + COMBAT.need,
                 COMBAT.kills >= COMBAT.need ? '서쪽 해안길 끝, 바다 위의 성으로' : (m.kind === 'golem' ? '얼음 골렘' : '촉수 괴물'), true);
      }
    }
    if (any) COMBAT.shake = 0.18;
    else strikeResource(fx, fz);           // nothing alive in reach: the blow lands on a tree or a rock
  }
  function hurtHero(m) {
    if (hero.hp <= 0) return;
    hero.hp = Math.max(0, hero.hp - m.K.dmg);
    var dx = hero.pos.x - m.root.position.x, dz = hero.pos.z - m.root.position.z, d = Math.max(0.01, Math.hypot(dx, dz));
    hero.vel.x += dx / d * 7; hero.vel.z += dz / d * 7; hero.vel.y = Math.max(hero.vel.y, 3); hero.onFloor = false;
    COMBAT.shake = 0.3; COMBAT.hurtFx = 1;
    updateHud();
    if (hero.hp <= 0) {
      showArea('쓰러졌다', '엘렌하르 마을에서 다시 눈을 뜬다', true);
      setTimeout(function () {
        var s = VILLAGE.data.spawn;
        hero.pos.set(s.x, groundAt(s.x, s.z), s.z); hero.vel.set(0, 0, 0); hero.hp = 100; updateHud();
      }, 1600);
    }
  }
  function updateMonsters(dt) {
    if (!LAND.ready || zone !== 'village') return;
    hero.pos.x = Math.max(-250, Math.min(250, hero.pos.x)); hero.pos.z = Math.max(-250, Math.min(250, hero.pos.z));   // the mapped square
    if (COMBAT.strike > 0 && (COMBAT.strike -= dt) <= 0) heroStrike();
    updateRes(dt);
    updatePickups(dt);
    for (var i = 0; i < MONSTERS.length; i++) {
      var m = MONSTERS[i], r = m.root, K = m.K;
      m.t += dt;
      if (m.state === 'dead') {                       // the remains settle into the ground
        m.timer += dt;
        if (m.timer > 2.5) r.position.y -= dt * 0.25;
        if (m.timer > 9) { r.parent && r.parent.remove(r); MONSTERS.splice(i--, 1); }
        continue;
      }
      var dxh = hero.pos.x - r.position.x, dzh = hero.pos.z - r.position.z, dh = Math.hypot(dxh, dzh);
      var moving = 0, toward = Math.atan2(dxh, dzh);
      if (m.flash > 0) {
        m.flash -= dt;
        var f = Math.max(0, m.flash) / 0.18;
        r.traverse(function (o) { if (o.isMesh) o.material.color.setRGB(1 + f * 2.2, 1 + f * 0.35, 1 + f * 0.3); });
      }
      if (m.knock.lengthSq() > 0.01) {
        r.position.x += m.knock.x * dt; r.position.z += m.knock.y * dt;
        m.knock.multiplyScalar(Math.max(0, 1 - 6 * dt));
      }
      if (m.cool > 0) m.cool -= dt;
      if (m.state === 'hurt') {
        if ((m.timer -= dt) <= 0) m.state = 'chase';
      } else if (m.state === 'attack') {
        m.timer += dt;
        m.heading += Math.atan2(Math.sin(toward - m.heading), Math.cos(toward - m.heading)) * Math.min(1, 3 * dt);
        if (m.timer >= K.windup && !m.struck) {
          m.struck = true;
          if (dh < K.reach + 0.6 && hero.hp > 0) hurtHero(m);
        }
        if (m.timer >= K.windup + 0.45) { m.state = 'chase'; m.cool = K.cool; }
      } else if (dh < 16 && hero.hp > 0 && zone === 'village' && !inVillage(hero.pos.x, hero.pos.z, 0)) {
        m.state = 'chase';                            // hunting Robin
        m.heading += Math.atan2(Math.sin(toward - m.heading), Math.cos(toward - m.heading)) * Math.min(1, 4 * dt);
        if (dh > K.reach) {
          var sp = K.run * (m.kind === 'horror' ? (0.8 + 0.2 * Math.sin(m.t * 3)) : 1);
          r.position.x += Math.sin(m.heading) * sp * dt; r.position.z += Math.cos(m.heading) * sp * dt;
          moving = 2;
        } else if (m.cool <= 0) { m.state = 'attack'; m.timer = 0; m.struck = false; }
      } else {
        m.state = 'roam';                             // wandering: the home drifts so the creatures range over the land
        if (m.wait > 0) m.wait -= dt;
        else {
          if (!m.target) {
            if (Math.random() < 0.35) {
              var a0 = Math.random() * 6.28, hx = m.home.x + Math.cos(a0) * 35, hz = m.home.y + Math.sin(a0) * 35;
              if (landOk(hx, hz)) m.home.set(hx, hz);
            }
            var a = Math.random() * 6.28, d = 6 + Math.random() * 18;
            m.target = new THREE.Vector2(m.home.x + Math.cos(a) * d, m.home.y + Math.sin(a) * d);
          }
          var tx = m.target.x - r.position.x, tz = m.target.y - r.position.z, td = Math.hypot(tx, tz);
          if (td < 0.8 || !landOk(m.target.x, m.target.y)) { m.target = null; m.wait = 1 + Math.random() * 4; }
          else {
            var want = Math.atan2(tx, tz), diff = Math.atan2(Math.sin(want - m.heading), Math.cos(want - m.heading));
            m.heading += diff * Math.min(1, 2 * dt);
            if (Math.abs(diff) < 0.7) { r.position.x += Math.sin(m.heading) * K.walk * dt; r.position.z += Math.cos(m.heading) * K.walk * dt; moving = 1; }
          }
        }
      }
      // keep out of the village and the sea
      if (inVillage(r.position.x, r.position.z, 3)) {      // the town wall keeps them out: back off, away from the village
        var ox = r.position.x - 4, oz = r.position.z - 18, od = Math.hypot(ox, oz) || 1;
        for (var it = 0; it < 60 && inVillage(r.position.x, r.position.z, 3); it++) { r.position.x += ox / od * 0.5; r.position.z += oz / od * 0.5; }
      }
      r.position.y = LAND.groundAt(r.position.x, r.position.z);
      r.rotation.y = m.heading;
      animateMonster(m, moving, dt, dh);
      if (dh < m.radius + RADIUS && dh > 1e-4) { hero.pos.x = r.position.x + dxh / dh * (m.radius + RADIUS); hero.pos.z = r.position.z + dzh / dh * (m.radius + RADIUS); }
    }
    updateDebris(dt);
    // the castle stays shut until enough monsters have fallen
    if (LAND.castle && COMBAT.kills < COMBAT.need) {
      var cd = Math.hypot(hero.pos.x - LAND.castle.x, hero.pos.z - LAND.castle.z);
      if (cd < 66) {
        var k = 66 / cd;
        hero.pos.x = LAND.castle.x + (hero.pos.x - LAND.castle.x) * k; hero.pos.z = LAND.castle.z + (hero.pos.z - LAND.castle.z) * k;
        if (clockNow - COMBAT.gateMsgAt > 5) { COMBAT.gateMsgAt = clockNow; showArea('다리가 막혀 있다', '몬스터를 ' + COMBAT.need + '마리 쓰러뜨려야 한다  (' + COMBAT.kills + ' / ' + COMBAT.need + ')', true); }
      }
    }
    if (COMBAT.shake > 0) {
      COMBAT.shake -= dt;
      camera.position.x += (Math.random() - 0.5) * COMBAT.shake * 0.5; camera.position.y += (Math.random() - 0.5) * COMBAT.shake * 0.5;
    }
    if (COMBAT.hurtFx > 0) { COMBAT.hurtFx = Math.max(0, COMBAT.hurtFx - dt * 2.2); HUD.hurt.style.opacity = COMBAT.hurtFx.toFixed(3); }
  }
  function animateMonster(m, moving, dt, dh) {
    m.phase += dt * (moving === 2 ? (m.kind === 'golem' ? 5.2 : 9) : moving ? (m.kind === 'golem' ? 3.2 : 5) : 1.2);
    var ph = m.phase, w = moving ? 1 : 0.15, atk = 0;
    if (m.state === 'attack') {                        // wind up, then strike
      var t = m.timer / m.K.windup;
      atk = t < 1 ? -t : Math.max(0, 1 - (m.timer - m.K.windup) / 0.45) * 1.6;
    }
    var hurt = m.state === 'hurt' ? Math.sin(m.timer * 20) * 0.15 : 0;
    if (m.kind === 'golem') {
      pose(m, 'thigh_l', Math.sin(ph) * 0.45 * w); pose(m, 'thigh_r', -Math.sin(ph) * 0.45 * w);
      pose(m, 'shin_l', Math.max(0, -Math.cos(ph)) * 0.5 * w); pose(m, 'shin_r', Math.max(0, Math.cos(ph)) * 0.5 * w);
      pose(m, 'arm_l', -Math.sin(ph) * 0.35 * w + atk * 1.4, 0, 0.05); pose(m, 'arm_r', Math.sin(ph) * 0.35 * w + atk * 1.4, 0, -0.05);
      pose(m, 'fore_l', -0.25 - Math.max(0, Math.sin(ph)) * 0.2 * w + atk * 0.4); pose(m, 'fore_r', -0.25 - Math.max(0, -Math.sin(ph)) * 0.2 * w + atk * 0.4);
      pose(m, 'torso', 0.04 + Math.sin(m.t * 1.3) * 0.025 - atk * 0.25 + hurt, Math.sin(ph) * 0.06 * w);
      pose(m, 'head', Math.sin(m.t * 0.7) * 0.05, dh < 16 ? 0 : Math.sin(m.t * 0.4) * 0.3);
      if (m.parts.pelvis) m.parts.pelvis.node.position.y = m.parts.pelvis.pos.y + Math.abs(Math.sin(ph)) * 0.05 * w;
    } else {
      for (var t2 = 0; t2 < 6; t2++) for (var k = 0; k < 5; k++) {
        var wave = Math.sin(m.t * (moving ? 6 : 1.6) - k * 0.7 + t2 * 1.05);
        pose(m, 'tent_' + t2 + '_' + k, wave * (0.1 + 0.05 * k) * (moving ? 1.8 : 1), 0, Math.cos(m.t * 1.1 + t2 + k * 0.5) * 0.08);
      }
      pose(m, 'arm_l', Math.sin(m.t * 1.7) * 0.15 - atk * 1.2, 0, Math.sin(m.t * 0.9) * 0.1); pose(m, 'arm_r', Math.sin(m.t * 1.7 + 1) * 0.15 - atk * 1.2, 0, -Math.sin(m.t * 0.9) * 0.1);
      pose(m, 'fin_l', 0, 0, Math.sin(m.t * (moving === 2 ? 6 : 2.3)) * 0.25); pose(m, 'fin_r', 0, 0, -Math.sin(m.t * (moving === 2 ? 6 : 2.3)) * 0.25);
      pose(m, 'head', Math.sin(m.t * 1.1) * 0.08 - atk * 0.3 + hurt, Math.sin(m.t * 0.6) * 0.2);
      if (m.parts.body) {
        m.parts.body.node.position.y = m.parts.body.pos.y + Math.sin(m.t * (moving ? 7 : 2)) * 0.04;
        m.parts.body.node.position.z = m.parts.body.pos.z + Math.max(0, atk) * 0.35;
      }
    }
  }
  function updateDebris(dt) {
    for (var i = COMBAT.debris.length - 1; i >= 0; i--) {
      var d = COMBAT.debris[i], o = d.obj;
      d.life -= dt;
      if (!d.rest) {
        d.vel.y -= 14 * dt;
        o.position.addScaledVector(d.vel, dt);
        o.rotation.x += d.spin.x * dt; o.rotation.y += d.spin.y * dt; o.rotation.z += d.spin.z * dt;
        var g = LAND.groundAt(o.position.x, o.position.z) + 0.08;
        if (o.position.y < g) {
          o.position.y = g;
          if (Math.abs(d.vel.y) < 1.2) { d.rest = d.vel.lengthSq() < 1; d.vel.multiplyScalar(0.5); d.vel.y = 0; }
          else { d.vel.y *= -0.32; d.vel.x *= 0.6; d.vel.z *= 0.6; }
          d.spin.multiplyScalar(0.55);
        }
      }
      if (d.life < 2) o.position.y -= dt * 0.2;
      if (d.life <= 0) { o.parent && o.parent.remove(o); COMBAT.debris.splice(i, 1); }
    }
    for (var j = COMBAT.sparks.length - 1; j >= 0; j--) {
      var s = COMBAT.sparks[j], arr = s.pts.geometry.attributes.position.array;
      s.life -= dt;
      for (var n = 0; n < s.vel.length; n++) {
        s.vel[n].y -= 12 * dt;
        arr[n * 3] += s.vel[n].x * dt; arr[n * 3 + 1] += s.vel[n].y * dt; arr[n * 3 + 2] += s.vel[n].z * dt;
      }
      s.pts.geometry.attributes.position.needsUpdate = true;
      s.pts.material.opacity = Math.max(0, s.life / 1.1);
      if (s.life <= 0) { scene.remove(s.pts); s.pts.geometry.dispose(); s.pts.material.dispose(); COMBAT.sparks.splice(j, 1); }
    }
  }

  // ---------------------------------------------------------------- the bag, the sword and the axe
  // Marta the merchant gives Robin a satchel (it hangs at his left hip and opens the item bar);
  // Oswin the guard then arms him with a sword and an axe. Keys 1-3 (or the bar) pick fist, sword, axe;
  // the chosen weapon sits in his right hand and swings with the Slash clip.
  var WEAPONS = [{ name: '주먹', dmg: 24, reach: 2.0, arc: 0.15, sever: 0.45 },
                 { name: '칼', dmg: 40, reach: 2.6, arc: 0.0, sever: 0.75, node: 'sword' },
                 { name: '도끼', dmg: 52, reach: 2.4, arc: 0.1, sever: 0.9, node: 'axe' }];
  var INV = { bag: false, weapons: false, held: 0, items: {}, bar: null };
  function attachItems(gltf) {
    // find each item's place on the bones by attaching it to a rest-pose copy of Robin, then hang that on the real one
    var rest = THREE.SkeletonUtils.clone(HERO_GLTF.scene), rb = {}, hb = {};
    rest.position.set(0, 0, 0); rest.rotation.set(0, 0, 0); rest.scale.set(1, 1, 1);
    rest.traverse(function (o) { if (o.isBone) { rb[o.name] = o; var r = HERO_REST[o.name]; if (r) { o.position.copy(r[0]); o.quaternion.copy(r[1]); o.scale.copy(r[2]); } } });
    rest.updateMatrixWorld(true);
    hero.root.traverse(function (o) { if (o.isBone) hb[o.name] = o; });
    gltf.scene.updateMatrixWorld(true);
    ['sword', 'axe', 'bag'].forEach(function (name) {
      var node = gltf.scene.getObjectByName(name), boneName = name === 'bag' ? 'Hips' : 'RightHand';
      if (!node || !rb[boneName] || !hb[boneName]) return;
      var local = new THREE.Matrix4().copy(rb[boneName].matrixWorld).invert().multiply(node.matrixWorld);
      var piece = node.clone(true);
      piece.matrixAutoUpdate = true;
      local.decompose(piece.position, piece.quaternion, piece.scale);
      piece.traverse(function (o) { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; o.frustumCulled = false; } });
      piece.visible = false;
      hb[boneName].add(piece);
      INV.items[name] = piece;
    });
    refreshItems();
  }
  function refreshItems() {
    if (INV.items.bag) INV.items.bag.visible = INV.bag;
    if (INV.items.sword) INV.items.sword.visible = INV.held === 1;
    if (INV.items.axe) INV.items.axe.visible = INV.held === 2;
    if (typeof renderBag === 'function') renderBag();
  }
  function hold(i) {
    if (!INV.bag || (i > 0 && !INV.weapons) || hero.attacking) return;
    INV.held = i; refreshItems();
  }
  function npcGifts(n, next) {
    var id = n.id;
    next = next || function () {};
    if (id === 'npc_marta' && !INV.bag) {
      say([{ who: id, text: '자, 맨손으로 다니지 말고 이 가죽 가방 가져가요. 사과 두 알도 넣어 뒀어요. 주운 건 여기 넣고, 배고프면 꺼내 먹어요.' }], function () {
        INV.bag = true; refreshItems(); addToBag('apple', 2);
        showArea('가죽 가방을 얻었다', 'I 키로 가방을 연다  ·  주운 것은 저절로 들어간다', true);
        next();
      });
    } else if (id === 'npc_oswin' && !INV.weapons) {
      if (!INV.bag) { say([{ who: id, text: '맨손으로 들판에 나가겠다고? 시장의 마르타에게 먼저 가방부터 얻어 오시오. 무기는 그다음이오.' }], next); return; }
      say([{ who: id, text: '가방이 있군. 좋소. 이 칼과 도끼를 가져가시오. 들판의 괴물들은 칼로 베고, 단단한 골렘은 도끼로 쪼개시오.' }], function () {
        INV.weapons = true; INV.held = 1; refreshItems(); showArea('칼과 도끼를 얻었다', '1 맨손 · 2 칼 · 3 도끼  ·  도끼로 나무를 베고 바위를 깬다', true);
        next();
      });
    } else if (id === 'npc_lina' && !QUEST.pears) {
      if (!INV.bag) { say([{ who: id, text: '들판에 나갈 거면 챙겨 줄 게 있는데… 어머, 가방이 없네요. 시장의 마르타한테 가방부터 얻어 와요.' }], next); return; }
      say([{ who: id, text: '들판에 나갈 거죠? 물푸레나무에서 딴 배 두 알이에요. 먹으면 한동안 팔에 힘이 붙어요.' }], function () {
        QUEST.pears = true; addToBag('pear', 2);
        next();
      });
    } else next();
  }

  // ---------------------------------------------------------------- the village quest
  // Elder Eldric asks Robin to hear the five villagers out; once he has, the elder sends him to the castle.
  // stage 0: listening   1: all heard, back to Eldric   2: sent to the castle   3: in the throne hall
  var QUEST = { stage: 0, heard: {}, need: ['npc_marta', 'npc_oswin', 'npc_bren', 'npc_lina', 'npc_toby'], pears: false };
  function eul(word) {                     // 을 after a closing consonant, 를 after a vowel
    var c = word.charCodeAt(word.length - 1) - 0xac00;
    return word + (c >= 0 && c < 11172 && c % 28 ? '을' : '를');
  }
  function heardAll() { return QUEST.need.every(function (id) { return QUEST.heard[id]; }); }
  function talkTo(n) {
    var id = n.id, d = n.def, lines, giving = false;
    n.talking = true;
    if (id === 'npc_eldric') {
      if (QUEST.stage >= 2) lines = d.mission;
      else if (heardAll()) { lines = d.give; giving = true; }
      else if (!n.met) lines = d.lines;
      else {
        var left = QUEST.need.filter(function (k) { return !QUEST.heard[k]; }).map(function (k) { return STORY.speakers[k]; });
        lines = d.again.map(function (l) { return { who: l.who, text: l.text.replace('{left}을(를)', eul(left.join(', '))) }; });
      }
    } else lines = !n.met ? d.lines : (QUEST.stage >= 2 && d.mission ? d.mission : d.again);
    say(lines, function () {
      n.talking = false; n.met = true;
      var fresh = QUEST.need.indexOf(id) >= 0 && !QUEST.heard[id];
      if (fresh) QUEST.heard[id] = true;
      npcGifts(n, function () {
        if (giving) {
          QUEST.stage = 2;
          showArea('새 임무 · 성으로', '북서쪽 해안의 둑길 · 다리 앞의 괴물을 쓰러뜨려 길을 열자', true);
        } else if (fresh && QUEST.stage === 0 && heardAll()) {
          QUEST.stage = 1;
          showArea('마을 사람들의 이야기를 모두 들었다', '장로 엘드릭에게 돌아가자', true);
        } else if (fresh) {
          var c = QUEST.need.filter(function (k) { return QUEST.heard[k]; }).length;
          toast(STORY.speakers[id] + '의 이야기를 들었다  ·  ' + c + ' / ' + QUEST.need.length);
        }
        setObjective();
      });
    });
  }
  function questObjective() {
    if (QUEST.stage === 0) return '마을 사람들의 이야기를 듣자  ·  ' + QUEST.need.filter(function (k) { return QUEST.heard[k]; }).length + ' / ' + QUEST.need.length;
    if (QUEST.stage === 1) return '장로 엘드릭에게 돌아가자';
    if (QUEST.stage === 2) return COMBAT.kills < COMBAT.need ? '다리를 막은 괴물을 쓰러뜨리자  ·  ' + COMBAT.kills + ' / ' + COMBAT.need
                                                             : '북서쪽 둑길을 건너 성의 왕좌의 대전으로 가자';
    return '왕좌의 대전에 닿았다  ·  엘렌하르의 왕자가 돌아왔다';
  }
  function renderQuestList() {
    var el = $('questlist');
    if (!el) {
      el = document.createElement('ul'); el.id = 'questlist';
      $('nameplate').appendChild(el);
    }
    el.hidden = zone !== 'village' || QUEST.stage > 0;
    if (el.hidden) return;
    el.innerHTML = QUEST.need.map(function (k) {
      return '<li' + (QUEST.heard[k] ? ' class="done"' : '') + '>' + STORY.speakers[k] + '</li>';
    }).join('');
  }
  function reachThrone() {
    if (QUEST.stage !== 2 || COMBAT.kills < COMBAT.need || talk.active) return;
    QUEST.stage = 3;
    hero.vel.x = hero.vel.z = 0;
    say(STORY.village.throne, function () {
      setObjective();
      showArea('왕좌의 대전', '엘렌하르의 마지막 왕자가 돌아왔다', true);
    });
  }
  window.addEventListener('keydown', function (e) {          // 1-3: what Robin holds
    var k = { Digit1: 0, Digit2: 1, Digit3: 2, Numpad1: 0, Numpad2: 1, Numpad3: 2 }[e.code];
    if (k == null || !started || talk.active) return;
    e.preventDefault(); hold(k);
  });

  // ---------------------------------------------------------------- trees and rocks that can be cut and broken
  // Every scattered tree and every rock small enough to split is registered here. A blow on one swaps its
  // instance for a free copy: trees sway back on a damped spring, and after enough axe blows topple under
  // their own weight (the torque grows as they lean), bounce once and leave a stump and logs; rocks shatter
  // into tumbling chunks and leave stones. Fists and the sword shake fruit out of the broadleaf trees.
  var RES = { grid: {}, live: [], tall: {} };
  var ZERO_M = new THREE.Matrix4().makeScale(0, 0, 0);
  function resAdd(kind, name, it, col, r, h) {
    var k = Math.floor(it[0] / LAND.cell) + ',' + Math.floor(it[2] / LAND.cell);
    (RES.grid[k] = RES.grid[k] || []).push({ kind: kind, name: name, it: it, col: col, r: r, h: h || 0, hits: 0, fruit: /oak|ash/.test(name) ? 3 : 0, obj: null, gone: false });
  }
  function resFind(fx, fz, reach) {
    var p = hero.pos, gx = Math.floor(p.x / LAND.cell), gz = Math.floor(p.z / LAND.cell), best = null, bd = 1e9;
    for (var dx = -1; dx <= 1; dx++) for (var dz = -1; dz <= 1; dz++) {
      var list = RES.grid[(gx + dx) + ',' + (gz + dz)];
      if (!list) continue;
      for (var i = 0; i < list.length; i++) {
        var r = list[i];
        if (r.gone) continue;
        var ex = r.it[0] - p.x, ez = r.it[2] - p.z, c = Math.hypot(ex, ez), d = c - r.r;
        if (d > reach || (ex * fx + ez * fz) / Math.max(c, 1e-3) < 0.25) continue;
        if (d < bd) { bd = d; best = r; }
      }
    }
    return best;
  }
  function hideInstance(it) {
    (it.refs || []).forEach(function (ref) { ref[0].setMatrixAt(ref[1], ZERO_M); ref[0].instanceMatrix.needsUpdate = true; });
  }
  function freeCopy(name, it, sink) {
    var proto = LAND.protos && LAND.protos[name];
    if (!proto) return null;
    var piv = new THREE.Group(), body = proto.clone(true);
    piv.position.set(it[0], it[1] - sink * it[4], it[2]);
    body.position.set(0, 0, 0); body.rotation.set(0, it[3], 0); body.scale.setScalar(it[4]);
    body.traverse(function (o) {
      o.matrixAutoUpdate = true;
      if (!o.isMesh) return;
      o.castShadow = true; o.receiveShadow = true;
      if (/^leaf_/.test(o.material.name)) o.customDepthMaterial = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking, map: o.material.map, alphaTest: 0.4 });
    });
    piv.add(body);
    LAND.group.add(piv);
    return piv;
  }
  function treeHeight(name) {
    if (RES.tall[name] == null) {
      var p = LAND.protos[name];
      RES.tall[name] = p ? new THREE.Box3().setFromObject(p).getSize(new THREE.Vector3()).y : 10;
    }
    return RES.tall[name];
  }
  function strikeResource(fx, fz) {
    if (zone !== 'village' || !LAND.ready) return;
    var W = WEAPONS[INV.held] || WEAPONS[0], r = resFind(fx, fz, W.reach + 0.3);
    if (!r) return;
    var it = r.it, axe = INV.held === 2, dx = it[0] - hero.pos.x, dz = it[2] - hero.pos.z, d = Math.hypot(dx, dz) || 1;
    dx /= d; dz /= d;
    _wp.set(it[0] - dx * r.r, hero.pos.y + 1.05, it[2] - dz * r.r);           // where the blow lands, on the bark or the stone
    if (r.kind === 'tree') {
      if (!r.obj) {
        r.obj = freeCopy(r.name, it, 0.15);
        if (!r.obj) return;
        hideInstance(it);
        r.tilt = 0; r.tiltV = 0; r.ax = new THREE.Vector3(1, 0, 0); r.fall = false; r.t = 0;
        RES.live.push(r);
      }
      r.ax.set(dz, 0, -dx);
      r.tiltV += (axe ? 0.2 : 0.12) / Math.max(0.7, it[4]);
      burst(_wp, axe ? 0x9a7a52 : 0x6b8a3e, axe ? 24 : 12, 3);
      COMBAT.shake = Math.max(COMBAT.shake, axe ? 0.16 : 0.1);
      if (r.fruit > 0 && (!axe || Math.random() < 0.4)) {                         // fruit shaken loose from the crown
        r.fruit--;
        var H = treeHeight(r.name) * it[4], a = Math.random() * 6.28, rr = 0.8 + Math.random() * 1.6;
        spawnPickup(/ash/.test(r.name) ? 'pear' : 'apple', it[0] + Math.cos(a) * rr, it[1] + H * 0.55, it[2] + Math.sin(a) * rr, Math.cos(a) * 0.6, 0, Math.sin(a) * 0.6);
      }
      if (axe && ++r.hits >= (it[4] > 1.15 ? 4 : 3)) {
        r.fall = true; r.fallV = 0.25; r.gone = true;
        // it comes to rest on the slope it falls onto, not at a fixed angle
        var reach = treeHeight(r.name) * it[4] * 0.55, drop = groundAt(it[0], it[2]) - groundAt(it[0] + dx * reach, it[2] + dz * reach);
        r.rest = Math.max(1.15, Math.min(1.95, Math.PI / 2 + Math.atan2(drop, reach) - 0.06));
        r.col.r = r.r * 0.85;                                                       // the stump stays solid
        var st = LAND.protos.tree_stump_01;
        if (st) {
          var sb = new THREE.Box3().setFromObject(st).getSize(new THREE.Vector3());
          var stump = freeCopy('tree_stump_01', [it[0], it[1], it[2], it[3] + 1.3, r.r * 2.5 / Math.max(sb.x, sb.z)], 0.05);
          if (stump) r.stump = stump;
        }
        showArea('나무가 쓰러진다', '통나무를 주워 가방에 넣자', true);
      } else if (!axe && !r.fruit && INV.weapons && clockNow - (RES.hintAt || -99) > 8) {
        RES.hintAt = clockNow; toast('나무는 도끼로 벨 수 있다  ·  3 도끼');
      }
    } else {
      burst(_wp, 0x8d8a83, axe ? 22 : 10, 3.4);
      burst(_wp, 0xffd9a0, 6, 5);                                                   // sparks off the edge
      COMBAT.shake = Math.max(COMBAT.shake, 0.14);
      if (!axe) { if (clockNow - (RES.hintAt || -99) > 8) { RES.hintAt = clockNow; toast(INV.weapons ? '바위는 도끼로 깨야 한다  ·  3 도끼' : '맨손으로는 바위를 깰 수 없다'); } return; }
      if (++r.hits < 3) return;
      r.gone = true; r.col.r = 0;
      hideInstance(it);
      var size = Math.max(r.r * 2.2, r.h), keys = ['rock_07', 'rock_09', 'stone_01'].filter(function (k) { return LAND.protos[k]; });
      for (var i = 0; i < 8 && keys.length; i++) {
        var proto = LAND.protos[keys[i % keys.length]], ch = proto.clone(true);
        var pb = new THREE.Box3().setFromObject(proto).getSize(new THREE.Vector3()), k = size * (0.2 + Math.random() * 0.18) / Math.max(pb.x, pb.y, pb.z);
        ch.position.set(it[0] + (Math.random() - 0.5) * size * 0.5, it[1] + r.h * (0.2 + Math.random() * 0.5), it[2] + (Math.random() - 0.5) * size * 0.5);
        ch.scale.setScalar(k); ch.rotation.set(Math.random() * 6, Math.random() * 6, Math.random() * 6);
        ch.traverse(function (o) { o.matrixAutoUpdate = true; if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
        LAND.group.add(ch);
        var a = Math.random() * 6.28;
        COMBAT.debris.push({ obj: ch, vel: new THREE.Vector3(Math.cos(a) * 3 + dx * 2.5, 2.5 + Math.random() * 3.5, Math.sin(a) * 3 + dz * 2.5),
                             spin: new THREE.Vector3((Math.random() - 0.5) * 8, (Math.random() - 0.5) * 8, (Math.random() - 0.5) * 8), life: 16, rest: false });
      }
      burst(_wp, 0x9a968e, 40, 5);
      COMBAT.shake = 0.32;
      for (var s = 0; s < 2 + (Math.random() < 0.5 ? 1 : 0); s++) {
        var b = Math.random() * 6.28;
        spawnPickup('stone', it[0], it[1] + r.h * 0.6, it[2], Math.cos(b) * 2.2, 3, Math.sin(b) * 2.2);
      }
    }
  }
  function updateRes(dt) {
    for (var i = RES.live.length - 1; i >= 0; i--) {
      var r = RES.live[i], o = r.obj;
      if (r.fall) {
        if (!r.down) {                                  // toppling: gravity's pull grows with the lean
          r.fallV += (0.9 + 6.0 * Math.sin(r.tilt)) * dt / Math.max(0.7, Math.sqrt(r.it[4]));
          r.tilt += r.fallV * dt;
          if (r.tilt >= r.rest) {
            r.tilt = r.rest;
            if (r.fallV > 0.8) { r.fallV *= -0.18; if (!r.hitGround) landTree(r); }
            else { r.down = true; r.fallV = 0; }
          }
        } else if ((r.t += dt) > 12) {                  // lies a good while, then settles into the ground
          o.position.y -= dt * 0.45;
          if (r.t > 17) { LAND.group.remove(o); RES.live.splice(i, 1); continue; }
        }
      } else {                                          // swaying back after a blow: a damped spring
        r.tiltV += (-r.tilt * 30 - r.tiltV * 3.2) * dt;
        r.tilt += r.tiltV * dt;
      }
      o.quaternion.setFromAxisAngle(r.ax, r.tilt);
    }
  }
  function landTree(r) {
    r.hitGround = true;
    var it = r.it, H = treeHeight(r.name) * it[4], fx = r.ax.z * -1, fz = r.ax.x;      // the fall direction, from the axis
    COMBAT.shake = Math.max(COMBAT.shake, 0.4);
    _wp.set(it[0] + fx * H * 0.65, it[1] + 0.8, it[2] + fz * H * 0.65);
    burst(_wp, 0x5f7f3a, 46, 6);
    burst(_wp, 0x8a7458, 20, 4);
    var n = it[4] > 1.0 ? 3 : 2;
    for (var k = 0; k < n; k++) {
      var t = 0.16 + k * 0.13, x = it[0] + fx * H * t, z = it[2] + fz * H * t;
      spawnPickup('log', x, groundAt(x, z) + 0.6, z, fz * (Math.random() - 0.5) * 2, 1.5, -fx * (Math.random() - 0.5) * 2);
    }
  }

  // ---------------------------------------------------------------- pickups, the bag and food
  var ITEMS = {
    apple: { name: '사과', size: 0.15, food: { hp: 15 }, note: '아삭하고 달다.' },
    pear: { name: '배', size: 0.16, food: { hp: 10, atk: 0.2, t: 30 }, note: '물이 많고 시원하다.' },
    meat: { name: '촉수 괴물 고기', size: 0.45, food: { hp: 30, atk: 0.5, t: 45 }, note: '질기고 비릿하지만, 먹으면 힘이 솟는다.' },
    core: { name: '얼음 골렘의 핵', size: 0.2, use: { atk: 0.4, t: 60 }, verb: '깨뜨리기', note: '차가운 빛이 맥박처럼 뛴다. 깨뜨리면 냉기가 팔에 스민다.' },
    log: { name: '통나무', size: 0.62, note: '도끼로 벤 나무 토막. 무언가를 지을 때 쓸 수 있을 것 같다.' },
    stone: { name: '돌', size: 0.24, note: '바위에서 떨어져 나온 단단한 돌 조각.' }
  };
  var PICK = { protos: {}, list: [], group: null, bag: {}, order: [], buff: null, icons: {}, sel: null, hintAt: -99, items: null };
  function protoOf(node, size) {
    var o = node.clone(true);
    o.position.set(0, 0, 0);
    o.traverse(function (c) { c.matrixAutoUpdate = true; if (c.isMesh) { c.castShadow = true; c.receiveShadow = true; } });
    o.updateMatrixWorld(true);
    var b = new THREE.Box3().setFromObject(o), c = b.getCenter(new THREE.Vector3()), sz = b.getSize(new THREE.Vector3());
    var k = size / Math.max(sz.x, sz.y, sz.z), g = new THREE.Group();
    o.scale.multiplyScalar(k);
    o.position.set(-c.x * k, -b.min.y * k, -c.z * k);
    g.add(o);
    return g;
  }
  function setupPickups(hq, food) {
    [['meat', hq], ['core', hq], ['log', hq], ['apple', food], ['pear', food]].forEach(function (e) {
      var n = e[1].scene.getObjectByName(e[0]);
      if (n) PICK.protos[e[0]] = protoOf(n, ITEMS[e[0]].size);
    });
    if (LAND.protos && LAND.protos.stone_01) PICK.protos.stone = protoOf(LAND.protos.stone_01, ITEMS.stone.size);
    var core = PICK.protos.core;
    if (core) core.traverse(function (o) { if (o.isMesh && o.material.emissiveMap) { o.material = o.material.clone(); o.material.emissiveIntensity = 2.2; } });
    PICK.group = new THREE.Group(); PICK.group.name = 'pickups';
    LAND.group.add(PICK.group);
    makeIcons();
  }
  function spawnPickup(type, x, y, z, vx, vy, vz) {
    var P = PICK.protos[type];
    if (!P || !PICK.group) return;
    var o = P.clone(true);
    o.position.set(x, y, z); o.rotation.y = Math.random() * 6.28;
    PICK.group.add(o);
    PICK.list.push({ type: type, obj: o, vel: new THREE.Vector3(vx || 0, vy || 0, vz || 0), rest: false, spin: (Math.random() - 0.5) * 7, fly: 0, age: 0, label: null });
    if (PICK.list.length > 90) dropPickup(0);
  }
  function dropPickup(i) {
    var p = PICK.list[i];
    PICK.group.remove(p.obj);
    if (p.label) p.label.remove();
    PICK.list.splice(i, 1);
  }
  var _pv = new THREE.Vector3();
  function updatePickups(dt) {
    if (!PICK.group) return;
    var W = canvas.clientWidth, Hh = canvas.clientHeight, chest = hero.pos.y + 1.1;
    for (var i = PICK.list.length - 1; i >= 0; i--) {
      var p = PICK.list[i], o = p.obj;
      p.age += dt;
      if (p.fly > 0) {                                   // into the satchel: a short arc to Robin's hip
        p.fly += dt / 0.32;
        var t = Math.min(1, p.fly);
        o.position.lerpVectors(p.from, _pv.set(hero.pos.x, hero.pos.y + 0.85, hero.pos.z), t);
        o.position.y += Math.sin(t * Math.PI) * 0.6;
        o.scale.setScalar(1 - t * 0.7);
        if (t >= 1) { addToBag(p.type, 1); dropPickup(i); }
        continue;
      }
      if (!p.rest) {
        p.vel.y -= 14 * dt;
        o.position.addScaledVector(p.vel, dt);
        o.rotation.y += p.spin * dt;
        var f = floorAt(o.position.x, o.position.z, o.position.y + 0.3);
        if (o.position.y < f) {
          o.position.y = f;
          if (Math.abs(p.vel.y) < 1.4) { p.vel.set(0, 0, 0); p.rest = true; }
          else { p.vel.y *= -0.3; p.vel.x *= 0.5; p.vel.z *= 0.5; p.spin *= 0.5; }
        }
      }
      var dx = o.position.x - hero.pos.x, dz = o.position.z - hero.pos.z, d = Math.hypot(dx, dz);
      if (p.age > 0.7 && d < 1.5 && Math.abs(o.position.y - hero.pos.y) < 1.8 && hero.hp > 0) {
        if (INV.bag) { p.fly = 1e-4; p.from = o.position.clone(); if (p.label) p.label.hidden = true; continue; }
        if (clockNow - PICK.hintAt > 7) { PICK.hintAt = clockNow; toast('가방이 없어 주울 수 없다  ·  마을 시장의 마르타를 찾아가자'); }
      }
      // a name over things within reach of the eye
      if (d < 9) {
        if (!p.label) { p.label = document.createElement('div'); p.label.className = 'picklabel'; p.label.textContent = ITEMS[p.type].name; stage.appendChild(p.label); }
        _pv.set(o.position.x, o.position.y + Math.max(0.35, ITEMS[p.type].size * 0.9), o.position.z).project(camera);
        var vis = _pv.z < 1 && Math.abs(_pv.x) < 1.1 && Math.abs(_pv.y) < 1.1 && !talk.active;
        p.label.hidden = !vis;
        if (vis) {
          p.label.style.transform = 'translate(' + ((_pv.x * 0.5 + 0.5) * W).toFixed(1) + 'px,' + ((-_pv.y * 0.5 + 0.5) * Hh).toFixed(1) + 'px) translate(-50%,-100%)';
          p.label.style.opacity = Math.min(1, (9 - d) / 2.5).toFixed(2);
        }
      } else if (p.label) p.label.hidden = true;
    }
    if (PICK.buff && PICK.buff.until <= clockNow) { PICK.buff = null; }
    if ((PICK.hudTick = (PICK.hudTick || 0) + dt) > 0.25) { PICK.hudTick = 0; updateBuffHud(); }
  }
  function addToBag(type, n) {
    PICK.bag[type] = (PICK.bag[type] || 0) + n;
    if (PICK.order.indexOf(type) < 0) PICK.order.push(type);
    toast(ITEMS[type].name + '  +' + n, PICK.icons[type]);
    renderBag();
  }
  function buffMult() { return PICK.buff && PICK.buff.until > clockNow ? 1 + PICK.buff.atk : 1; }
  function useItem(type) {
    var D = ITEMS[type], e = D && (D.food || D.use);
    if (!e || !PICK.bag[type] || hero.hp <= 0) return;
    PICK.bag[type]--;
    if (!PICK.bag[type]) { PICK.order.splice(PICK.order.indexOf(type), 1); if (PICK.sel === type) PICK.sel = null; }
    var said = [];
    if (e.hp) { var before = hero.hp; hero.hp = Math.min(100, hero.hp + e.hp); said.push('체력 +' + Math.round(hero.hp - before)); }
    if (e.atk) {
      var b = PICK.buff && PICK.buff.until > clockNow ? PICK.buff : null;
      PICK.buff = { atk: Math.max(e.atk, b ? b.atk : 0), until: Math.max(clockNow + e.t, b ? b.until : 0), cold: type === 'core' || (b && b.cold) };
      said.push('공격력 +' + Math.round(e.atk * 100) + '%  ' + e.t + '초');
    }
    _wp.set(hero.pos.x, hero.pos.y + 1.2, hero.pos.z);
    burst(_wp, type === 'core' ? 0x9fdcff : 0xffc27a, 26, 2.2);
    toast(eul(D.name) + (D.food ? ' 먹었다' : ' 깨뜨렸다') + '  ·  ' + said.join('  ·  '), PICK.icons[type]);
    updateHud(); updateBuffHud(); renderBag();
  }
  function dropLoot(m) {
    var p = m.root.position, golem = m.kind === 'golem', n = golem ? 1 : 1 + (Math.random() < 0.45 ? 1 : 0);
    for (var i = 0; i < n; i++) {
      var a = Math.random() * 6.28;
      spawnPickup(golem ? 'core' : 'meat', p.x, p.y + (golem ? 1.6 : 1.0), p.z, Math.cos(a) * 1.8, 3.8, Math.sin(a) * 1.8);
    }
    if (golem) for (var j = 0; j < 2; j++) { var b = Math.random() * 6.28; spawnPickup('stone', p.x, p.y + 1.2, p.z, Math.cos(b) * 2.4, 3.2, Math.sin(b) * 2.4); }
  }

  // icons: each item rendered once, off screen, with the same sky light as the world
  function iconOf(node, prep) {
    var S = 192, rt = new THREE.WebGLRenderTarget(S, S);
    rt.texture.encoding = THREE.sRGBEncoding;
    var sc = new THREE.Scene(), cam = new THREE.PerspectiveCamera(26, 1, 0.01, 60);
    sc.environment = LAND.envTex || null;
    sc.add(new THREE.HemisphereLight(0xfff1de, 0x40372b, 0.7));
    var key = new THREE.DirectionalLight(0xfff4e8, 2.4); key.position.set(2.2, 3.4, 2.6); sc.add(key);
    var rim = new THREE.DirectionalLight(0xbfd6ff, 1.1); rim.position.set(-2.5, 1.2, -2); sc.add(rim);
    var o = node.clone(true), holder = new THREE.Group();
    o.position.set(0, 0, 0); o.scale.set(1, 1, 1);
    holder.add(o);
    if (prep) prep(o, holder);
    sc.add(holder);
    holder.updateMatrixWorld(true);
    var b = new THREE.Box3().setFromObject(holder), c = b.getCenter(new THREE.Vector3()), rad = b.getSize(new THREE.Vector3()).length() / 2;
    var dist = rad / Math.sin(THREE.MathUtils.degToRad(13)) * 1.04;
    cam.position.set(c.x + dist * 0.2, c.y + dist * 0.45, c.z + dist * 0.87); cam.lookAt(c);
    var prevT = renderer.getRenderTarget(), cc = renderer.getClearColor(new THREE.Color()), ca = renderer.getClearAlpha();
    renderer.setRenderTarget(rt); renderer.setClearColor(0x000000, 0); renderer.clear(); renderer.render(sc, cam);
    var px = new Uint8Array(S * S * 4);
    renderer.readRenderTargetPixels(rt, 0, 0, S, S, px);
    renderer.setRenderTarget(prevT); renderer.setClearColor(cc, ca);
    rt.dispose();
    var cv = document.createElement('canvas'); cv.width = cv.height = S;
    var cx = cv.getContext('2d'), im = cx.createImageData(S, S);
    for (var y = 0; y < S; y++) im.data.set(px.subarray((S - 1 - y) * S * 4, (S - y) * S * 4), y * S * 4);
    cx.putImageData(im, 0, 0);
    var out = document.createElement('canvas'); out.width = out.height = 96;
    var ox = out.getContext('2d'); ox.imageSmoothingQuality = 'high'; ox.drawImage(cv, 0, 0, 96, 96);
    var cover = 0;
    for (var a = 3; a < px.length; a += 4) if (px[a] > 40) cover++;
    return { url: out.toDataURL('image/png'), cover: cover };
  }
  function makeIcons() {
    if (!PICK.group || !PICK.items) return;          // both the pickups and Robin's gear must be loaded
    Object.keys(PICK.protos).forEach(function (k) { PICK.icons[k] = iconOf(PICK.protos[k], k === 'log' ? function (o, h) { h.rotation.y = 0.6; } : null).url; });
    ['sword', 'axe'].forEach(function (k) {
      var n = PICK.items.scene.getObjectByName(k);
      if (!n) return;
      // stand the blade up the diagonal, turned whichever way shows its broad face to the camera
      var best = null;
      [0, Math.PI / 2].forEach(function (spin) {
        var ic = iconOf(n, function (o) { o.quaternion.identity(); o.rotation.set(-1.75, spin, -0.78, 'ZYX'); });
        if (!best || ic.cover > best.cover) best = ic;
      });
      PICK.icons[k] = best.url;
    });
    renderBag();
  }

  // the satchel panel (I / B, or the button under Robin's health): what he holds, what he carries
  (function bagPanel() {
    var css = document.createElement('style');
    css.textContent = [
      '#bagbtn{position:absolute;left:max(16px,env(safe-area-inset-left,0px));top:168px;display:flex;align-items:center;gap:8px;padding:6px 10px;',
      '  font:600 12px var(--font-ui);letter-spacing:.08em;color:var(--ink);cursor:pointer;pointer-events:auto}',
      '#bagbtn kbd{font-size:10px;padding:1px 5px}',
      '#bagbtn .held{color:var(--ink-dim);font-weight:500}',
      '#bagpanel{position:absolute;right:max(16px,env(safe-area-inset-right,0px));top:50%;transform:translateY(-50%);width:340px;max-width:calc(100vw - 32px);',
      '  padding:16px 16px 14px;pointer-events:auto;max-height:calc(100% - 120px);overflow:auto}',
      '#bagpanel .bp-head{display:flex;align-items:baseline;justify-content:space-between;margin-bottom:12px}',
      '#bagpanel .bp-title{font-family:var(--font-display);font-size:22px;font-weight:700;letter-spacing:.2em;color:var(--ink)}',
      '#bagpanel .bp-x{font:500 12px var(--font-ui);color:var(--ink-dim);background:none;border:0;cursor:pointer;padding:4px}',
      '#bagpanel .bp-sec{font-size:11px;letter-spacing:.16em;color:var(--ink-dim);margin:12px 0 7px}',
      '#bagpanel .bp-hands{display:grid;grid-template-columns:repeat(3,1fr);gap:7px}',
      '#bagpanel .slot{position:relative;aspect-ratio:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:2px;cursor:pointer;',
      '  font:600 12px var(--font-ui);color:var(--ink);background:rgba(243,239,228,.05);border:1px solid var(--panel-line);border-radius:3px;padding:4px}',
      '#bagpanel .slot img{width:72%;height:auto;pointer-events:none;filter:drop-shadow(0 2px 3px rgba(0,0,0,.45))}',
      '#bagpanel .bp-hands .slot{aspect-ratio:auto;height:74px}',
      '#bagpanel .bp-hands .slot img{width:46px}',
      '#bagpanel .slot kbd{position:absolute;left:4px;top:4px;font-size:9px;padding:0 4px}',
      '#bagpanel .slot .n{position:absolute;right:5px;bottom:3px;font-size:12px;color:var(--ink);text-shadow:0 1px 2px #000}',
      '#bagpanel .slot.on{border-color:var(--ember);background:rgba(243,169,95,.14);box-shadow:inset 0 0 0 1px var(--ember)}',
      '#bagpanel .slot:disabled,#bagpanel .slot.empty{opacity:.35;cursor:default}',
      '#bagpanel .slot.empty{opacity:.5}',
      '#bagpanel .bp-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:7px}',
      '#bagpanel .bp-detail{margin-top:12px;min-height:84px;padding-top:11px;border-top:1px solid var(--panel-line)}',
      '#bagpanel .bp-name{font-size:15px;font-weight:700;color:var(--ink);margin-bottom:4px}',
      '#bagpanel .bp-note{font-size:13px;line-height:1.5;color:var(--ink-dim)}',
      '#bagpanel .bp-fx{font-size:12px;color:var(--ember);margin-top:5px}',
      '#bagpanel .bp-use{margin-top:9px;font:600 13px var(--font-ui);color:#1b1410;background:var(--ember);border:0;border-radius:3px;padding:7px 16px;cursor:pointer}',
      '#bagpanel .bp-empty{font-size:13px;color:var(--ink-dim);line-height:1.5}',
      '#stage.talking #bagpanel,#stage.talking #bagbtn,#stage.talking .picklabel{display:none}',
      '.picklabel{position:absolute;left:0;top:0;pointer-events:none;white-space:nowrap;font:600 12px var(--font-ui);letter-spacing:.06em;color:#f3efe4;',
      '  padding:3px 8px;border-radius:2px;background:rgba(11,14,12,.55);text-shadow:0 1px 2px rgba(0,0,0,.8)}',
      '#toasts{position:absolute;left:50%;bottom:92px;transform:translateX(-50%);display:flex;flex-direction:column;align-items:center;gap:6px;pointer-events:none}',
      '#toasts .t{display:flex;align-items:center;gap:8px;font:600 13px var(--font-ui);color:var(--ink);padding:6px 12px;border-radius:3px;',
      '  background:rgba(11,14,12,.66);border-left:2px solid var(--ember);animation:tIn .25s ease-out;transition:opacity .6s}',
      '#toasts .t img{width:26px;height:26px}',
      '@keyframes tIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}',
      '#hud-hp .buff{margin-top:6px;font-size:11px;letter-spacing:.06em;color:#f3a95f}',
      '#hud-hp .buff.cold{color:#9fdcff}',
      '@media (max-width:560px){#bagpanel{top:auto;bottom:12px;transform:none;right:16px;left:16px;width:auto}#toasts{bottom:150px}}'].join('\n');
    document.head.appendChild(css);
    var btn = document.createElement('button'); btn.type = 'button'; btn.id = 'bagbtn'; btn.className = 'panel'; btn.hidden = true;
    btn.innerHTML = '<span>가방</span><kbd>I</kbd><span class="held"></span>';
    var panel = document.createElement('div'); panel.id = 'bagpanel'; panel.className = 'panel'; panel.hidden = true;
    panel.setAttribute('role', 'dialog'); panel.setAttribute('aria-label', '가방');
    panel.innerHTML = '<div class="bp-head"><span class="bp-title">가방</span><button type="button" class="bp-x">닫기 <kbd>I</kbd></button></div>' +
      '<div class="bp-sec">손에 들 것</div><div class="bp-hands"></div><div class="bp-sec">소지품</div><div class="bp-grid"></div><div class="bp-detail"></div>';
    var toasts = document.createElement('div'); toasts.id = 'toasts';
    stage.appendChild(btn); stage.appendChild(panel); stage.appendChild(toasts);
    var stop = function (e) { e.stopPropagation(); };
    ['pointerdown', 'mousedown', 'touchstart', 'click'].forEach(function (ev) { panel.addEventListener(ev, stop); btn.addEventListener(ev, stop); });
    btn.addEventListener('click', function () { toggleBag(); });
    panel.querySelector('.bp-x').addEventListener('click', function () { toggleBag(false); });
    panel.addEventListener('click', function (e) {
      var s = e.target.closest('[data-hand]');
      if (s) { hold(+s.dataset.hand); return; }
      s = e.target.closest('[data-item]');
      if (s) { PICK.sel = s.dataset.item; renderBag(); return; }
      if (e.target.closest('.bp-use') && PICK.sel) useItem(PICK.sel);
    });
    window.addEventListener('keydown', function (e) {
      if (!started || talk.active) return;
      if (e.code === 'KeyI' || e.code === 'KeyB') { e.preventDefault(); toggleBag(); }
      else if (e.code === 'Escape' && !panel.hidden) { e.preventDefault(); toggleBag(false); }
    });
    PICK.ui = { btn: btn, panel: panel, toasts: toasts };
  })();
  function toggleBag(open) {
    var p = PICK.ui.panel;
    if (!INV.bag) { toast('아직 가방이 없다  ·  마을 시장의 마르타를 찾아가자'); return; }
    p.hidden = open == null ? !p.hidden : !open;
    if (!p.hidden) renderBag();
    else canvas.focus();
  }
  function toast(text, icon) {
    var t = document.createElement('div'); t.className = 't';
    t.innerHTML = (icon ? '<img alt="" src="' + icon + '">' : '') + '<span></span>';
    t.lastChild.textContent = text;
    PICK.ui.toasts.appendChild(t);
    while (PICK.ui.toasts.children.length > 4) PICK.ui.toasts.firstChild.remove();
    setTimeout(function () { t.style.opacity = '0'; }, 2600);
    setTimeout(function () { t.remove(); }, 3300);
  }
  function renderBag() {
    if (!PICK.ui) return;
    var ui = PICK.ui, W = WEAPONS[INV.held] || WEAPONS[0];
    ui.btn.hidden = !INV.bag;
    ui.btn.querySelector('.held').textContent = '· ' + W.name;
    if (ui.panel.hidden) return;
    ui.panel.querySelector('.bp-hands').innerHTML = WEAPONS.map(function (w, i) {
      var ic = PICK.icons[w.node];
      return '<button type="button" class="slot' + (i === INV.held ? ' on' : '') + '" data-hand="' + i + '"' + (i > 0 && !INV.weapons ? ' disabled' : '') + '>' +
        '<kbd>' + (i + 1) + '</kbd>' + (ic ? '<img alt="" src="' + ic + '">' : '') + '<span>' + w.name + '</span></button>';
    }).join('');
    var cells = PICK.order.map(function (k) {
      var ic = PICK.icons[k];
      return '<button type="button" class="slot' + (PICK.sel === k ? ' on' : '') + '" data-item="' + k + '" title="' + ITEMS[k].name + '">' +
        (ic ? '<img alt="" src="' + ic + '">' : '<span>' + ITEMS[k].name + '</span>') + '<span class="n">' + PICK.bag[k] + '</span></button>';
    });
    for (var e = cells.length; e < 12; e++) cells.push('<div class="slot empty"></div>');
    ui.panel.querySelector('.bp-grid').innerHTML = cells.join('');
    var det = ui.panel.querySelector('.bp-detail'), k = PICK.sel;
    if (!k || !PICK.bag[k]) {
      det.innerHTML = '<div class="bp-empty">' + (PICK.order.length ? '물건을 골라 살펴본다.' : '아직 아무것도 없다. 열매를 흔들어 떨구고, 나무를 베고, 바위를 깨고, 괴물을 쓰러뜨려 얻는다.') + '</div>';
      return;
    }
    var D = ITEMS[k], e2 = D.food || D.use, fx = [];
    if (e2 && e2.hp) fx.push('체력 +' + e2.hp);
    if (e2 && e2.atk) fx.push('공격력 +' + Math.round(e2.atk * 100) + '% · ' + e2.t + '초');
    det.innerHTML = '<div class="bp-name"></div><div class="bp-note"></div>' + (fx.length ? '<div class="bp-fx">' + fx.join('  ·  ') + '</div>' : '') +
      (e2 ? '<button type="button" class="bp-use">' + (D.food ? '먹기' : D.verb) + '</button>' : '');
    det.querySelector('.bp-name').textContent = D.name + '  ×' + PICK.bag[k];
    det.querySelector('.bp-note').textContent = D.note;
  }
  function updateBuffHud() {
    var el = $('hud-buff');
    if (!el) return;
    var b = PICK.buff;
    el.hidden = !b;
    if (b) { el.textContent = (b.cold ? '냉기 ' : '힘 ') + '공격력 +' + Math.round(b.atk * 100) + '%  ·  ' + Math.ceil(b.until - clockNow) + '초'; el.classList.toggle('cold', !!b.cold); }
  }

  // ---------------------------------------------------------------- HUD: health, area banner
  var HUD = (function () {
    var css = document.createElement('style');
    css.textContent = [
      '#hud-hp{position:absolute;left:max(16px,env(safe-area-inset-left,0px));top:96px;width:180px;padding:7px 10px 8px;pointer-events:none}',
      '#hud-hp .lbl{display:flex;justify-content:space-between;font-size:11px;letter-spacing:.12em;color:var(--ink-dim);margin-bottom:5px}',
      '#hud-hp .bar{height:6px;border-radius:3px;background:rgba(243,239,228,.12);overflow:hidden}',
      '#hud-hp .fill{height:100%;width:100%;background:linear-gradient(90deg,#b8442f,#e0794e);transition:width .25s ease-out}',
      '#hud-hurt{position:absolute;inset:0;pointer-events:none;opacity:0;background:radial-gradient(circle at 50% 50%,rgba(140,0,0,0) 45%,rgba(140,0,0,.55) 100%)}',
      '#area{position:absolute;left:max(16px,env(safe-area-inset-left,0px));bottom:262px;pointer-events:none;opacity:0;transform:translateY(8px);',
      '  transition:opacity 1.1s ease,transform 1.1s ease;max-width:min(86vw,560px);padding:14px 64px 15px 20px;',
      '  border-left:2px solid var(--ember);background:linear-gradient(90deg,rgba(11,14,12,.72) 0%,rgba(11,14,12,.45) 60%,rgba(11,14,12,0) 100%)}',
      '#area.on{opacity:1;transform:none}',
      '#area .name{font-family:var(--font-display);font-size:34px;font-weight:700;letter-spacing:.22em;color:var(--ink);',
      '  text-shadow:0 2px 14px rgba(0,0,0,.65),0 0 2px rgba(0,0,0,.8)}',
      '#area .rule{height:1px;width:0;margin:9px 0 8px;background:linear-gradient(90deg,var(--ember),rgba(243,169,95,0));transition:width 1.6s ease .25s}',
      '#area.on .rule{width:260px}',
      '#area .sub{font-size:14px;letter-spacing:.18em;color:var(--ink-dim);text-shadow:0 1px 8px rgba(0,0,0,.7)}',
      '@media (max-width:560px){#area .name{font-size:25px}#area{bottom:170px}}'].join('\n');
    document.head.appendChild(css);
    var hp = document.createElement('div'); hp.id = 'hud-hp'; hp.className = 'panel'; hp.hidden = true;
    hp.innerHTML = '<div class="lbl"><span>로빈</span><span id="hud-hp-n">100</span></div><div class="bar"><div class="fill" id="hud-hp-f"></div></div><div class="buff" id="hud-buff" hidden></div>';
    var hurt = document.createElement('div'); hurt.id = 'hud-hurt';
    var area = document.createElement('div'); area.id = 'area'; area.setAttribute('aria-live', 'polite');
    area.innerHTML = '<div class="name"></div><div class="rule"></div><div class="sub"></div>';
    stage.appendChild(hurt); stage.appendChild(hp); stage.appendChild(area);
    return { hp: hp, hurt: hurt, area: area, current: '', timer: null };
  })();
  (function hudColumn() {
    // the nameplate (with the quest), Robin's health and the satchel button stack down the left, so none overlap
    var css = document.createElement('style');
    css.textContent = [
      '.hudcol{display:flex;flex-direction:column;align-items:flex-start;gap:8px;min-width:0;max-width:min(100%,420px)}',
      '.hudcol #hud-hp,.hudcol #bagbtn{position:static}',
      '#questlist{list-style:none;margin:2px 0 0;padding:0;display:flex;flex-wrap:wrap;gap:5px}',
      '#questlist li{font-size:12px;letter-spacing:.04em;color:var(--ink-dim);padding:2px 8px 3px;border:1px solid var(--panel-line);border-radius:2px}',
      '#questlist li.done{color:var(--ink);border-color:rgba(243,169,95,.55);background:rgba(243,169,95,.12)}',
      '#questlist li.done::before{content:"\\2713";color:var(--ember);margin-right:5px;font-weight:700}'].join('\n');
    document.head.appendChild(css);
    var top = document.querySelector('.overlay.top'), np = $('nameplate'), col = document.createElement('div');
    col.className = 'hudcol';
    top.insertBefore(col, np);
    col.appendChild(np); col.appendChild(HUD.hp); col.appendChild(PICK.ui.btn);
  })();
  hero.hp = 100;
  function updateHud() {
    HUD.hp.hidden = zone !== 'village';
    $('hud-hp-n').textContent = Math.ceil(hero.hp);
    $('hud-hp-f').style.width = hero.hp + '%';
  }
  function showArea(name, sub, force) {
    if (!force && HUD.current === name) return;
    if (!force) HUD.current = name;
    var a = HUD.area;
    a.querySelector('.name').textContent = name;
    a.querySelector('.sub').textContent = sub || '';
    a.classList.remove('on'); void a.offsetWidth; a.classList.add('on');
    clearTimeout(HUD.timer);
    HUD.timer = setTimeout(function () { a.classList.remove('on'); }, 4200);
  }
  var AREA_TICK = 0;
  function updateArea() {
    if (!started || hero.cutscene || (AREA_TICK++ % 10)) return;
    var x = hero.pos.x, z = hero.pos.z, name, sub;
    var inRoom = zone === 'village' ? roomAt(x, hero.pos.y + 0.5, z, -0.05) : null;
    if (zone === 'forest') { name = '메갈리스 유적'; sub = '잊힌 왕국의 돌 고리'; }
    else if (inRoom) { name = inRoom.name; sub = '엘렌하르 마을'; }
    else if (LAND.castle && Math.hypot(x - LAND.castle.x, z - LAND.castle.z) < 30) {
      var k = LAND.castle.keep, kx = LAND.castle.x + k.x, kz = LAND.castle.z + k.z;
      if (Math.abs(x - kx) < k.w / 2 && Math.abs(z - kz) < k.d / 2 && hero.pos.y > LAND.castle.y + 0.8) { name = '왕좌의 대전'; sub = '엘렌하르 성'; reachThrone(); }
      else { name = '엘렌하르 성'; sub = '바다 위의 옛 왕성'; }
    }
    else if (LAND.castle && Math.hypot(x - LAND.castle.x, z - LAND.castle.z) < 75) { name = '해안 둑길'; sub = '성으로 이어지는 돌다리'; }
    else if (inVillage(x, z, 0)) { name = '엘렌하르 마을'; sub = '성벽 안, 바다와 언덕 사이의 마을'; }
    else if (x < -95) { name = '고분의 숲'; sub = '오래된 참나무와 전나무 숲'; }
    else if (z > 150) { name = '남쪽 해안'; sub = '바람이 머무는 언덕 끝'; }
    else { name = '바람언덕 들판'; sub = '엘렌하르를 둘러싼 풀밭'; }
    showArea(name, sub);
    updateHud();
  }

  function loadCastleAndMonsters() {
    var getJson = function (u) { return fetch(u).then(function (r) { if (!r.ok) throw new Error(u + ' ' + r.status); return r.json(); }); };
    var whenLand = function (fn) { (function wait() { if (LAND.ready) fn(); else setTimeout(wait, 200); })(); };
    fetchGz('items.b64.txt').then(parseBytes).then(function (g) {
      (function wait() { if (hero.root && HERO_REST) { attachItems(g); PICK.items = g; makeIcons(); } else setTimeout(wait, 200); })();
    }).catch(function (err) { console.error('items', err); });
    Promise.all([fetchGz('pickups_hq.b64.txt').then(parseBytes), fetchGz('pickups_food.b64.txt').then(parseBytes)]).then(function (r) {
      whenLand(function () { setupPickups(r[0], r[1]); });
    }).catch(function (err) { console.error('pickups', err); });
    Promise.all([fetchGz('castle.b64.txt').then(parseBytes), getJson('castle.json')]).then(function (r) {
      whenLand(function () { buildCastle(r[0], r[1]); });
    }).catch(function (err) { console.error('castle', err); });
    Promise.all([fetchGz('horror.b64.txt').then(parseBytes), fetchGz('golem.b64.txt').then(parseBytes)]).then(function (r) {
      (function wait() {          // after the land and the castle, so they keep off its island
        if (LAND.ready && LAND.castle) { spawnMonsters('horror', r[0]); spawnMonsters('golem', r[1]); } else setTimeout(wait, 250);
      })();
    }).catch(function (err) { console.error('monsters', err); });
  }
  function loadLand() {
    var getJson = function (u) { return fetch(u).then(function (r) { if (!r.ok) throw new Error(u + ' ' + r.status); return r.json(); }); };
    var splatPixels = new Promise(function (resolve, reject) {
      var img = new Image();
      img.onload = function () {
        var cv = document.createElement('canvas'); cv.width = img.width; cv.height = img.height;
        var cx = cv.getContext('2d'); cx.drawImage(img, 0, 0);
        resolve(cx.getImageData(0, 0, img.width, img.height).data);
      };
      img.onerror = reject; img.src = 'land_splat.png';
    });
    if (THREE.RGBELoader) {               // the HDR travels as base64 text (artifacts don't serve .hdr)
      fetch('sky/sky_env.hdr.txt').then(function (r) { return r.text(); }).then(function (t) {
        var bin = atob(t.trim()), buf = new Uint8Array(bin.length);
        for (var i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
        var d = new THREE.RGBELoader().setDataType(THREE.UnsignedByteType).parse(buf.buffer);
        var hdr = new THREE.DataTexture(d.data, d.width, d.height, d.format, d.type);
        hdr.encoding = THREE.RGBEEncoding; hdr.flipY = true; hdr.generateMipmaps = false;
        hdr.minFilter = hdr.magFilter = THREE.NearestFilter; hdr.mapping = THREE.EquirectangularReflectionMapping;
        hdr.needsUpdate = true;
        var pm = new THREE.PMREMGenerator(renderer);
        LAND.envTex = pm.fromEquirectangular(hdr).texture;
        pm.dispose(); hdr.dispose();
        if ((LAND.ready && zone === 'village') || (FL.ready && zone === 'forest')) scene.environment = LAND.envTex;
      }).catch(function (err) { console.error('sky', err); });
    }
    var glb = function (n) { return fetchGz(n + '.b64.txt').then(parseBytes); };
    Promise.all([getJson('land.json'), glb('trees_broad'), glb('trees_fir'), glb('props_hard'), glb('props_soft'),
                 getJson('props.json'), getJson('sky/sky.json'), splatPixels]).then(function (r) {
      // gather both halves of each pack under one scene
      [r[2], r[4]].forEach(function (g, i) { var into = (i ? r[3] : r[1]).scene; g.scene.children.slice().forEach(function (c) { into.add(c); }); });
      buildLand(r[0], r[1], r[3], r[5], r[6], r[7]);
    }).catch(function (err) { console.error('land', err); });
  }

  // ---------------------------------------------------------------- developer start menu
  var devMenu = $('devstart'), started = false;
  function skipIntro() {
    $('veil').classList.add('open');
    intro.woke = true; hero.cutscene = false;
    cam.distance = cam.goal = 5.6; cam.pitch = 0.36;
    finishIntro();
  }
  var STARTS = {
    intro: function () { startIntro(); },
    ruins: function () { skipIntro(); },
    door: function () {
      skipIntro();
      foundCount = 3; found[0] = found[1] = found[2] = true; setObjective();
      // beside the altar, close enough to the door that F speaks to the door
      var d = POINTS.filter(function (x) { return x.id === 'door'; })[0];
      var sx = d.x + 1.7, sz = d.z + 0.9;
      hero.pos.set(sx, groundAt(sx, sz), sz); hero.heading = Math.atan2(d.x - sx, d.z - sz);
      cam.yaw = 0.9; focus.set(hero.pos.x, hero.pos.y + 1.3, hero.pos.z);
    },
    village: function () {
      $('veil').classList.add('open');
      intro.woke = true; hero.cutscene = false; finished = true;
      $('nameplate').hidden = false; $('legend').hidden = false;
      if (!(VILLAGE && VILLAGE.ready)) { $('village-wait').hidden = false; devMenu.hidden = false; }
      whenVillage(function () { devMenu.hidden = true; enterVillage({ quiet: true }); });
    }
  };
  function begin(which) {
    if (started) return;
    started = true;
    if (which !== 'village') devMenu.hidden = true;     // the village start closes it once the village is ready
    STARTS[which]();
    startMusic();
    if (zone === 'forest') forestLights();
    canvas.focus();
  }
  [['start-intro', 'intro'], ['start-ruins', 'ruins'], ['start-door', 'door'], ['start-village', 'village']].forEach(function (b) {
    $(b[0]).addEventListener('click', function () { begin(b[1]); });
  });
  window.addEventListener('keydown', function (e) {
    if (started || devMenu.hidden) return;
    var pick = { Digit1: 'intro', Digit2: 'ruins', Digit3: 'door', Digit4: 'village', Numpad1: 'intro', Numpad2: 'ruins', Numpad3: 'door', Numpad4: 'village' }[e.code];
    if (pick) { e.preventDefault(); begin(pick); }
  });

  // ---------------------------------------------------------------- start up
  function fetchGz(url) {
    return fetch(url).then(function (r) { if (!r.ok) throw new Error(url + ' ' + r.status); return r.text(); }).then(function (t) {
      var bin = atob(t.trim()), out = new Uint8Array(bin.length);
      for (var i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
      return fflate.gunzipSync(out);
    });
  }
  function parseBytes(bytes, now) {
    return new Promise(function (resolve, reject) {
      (function go() {      // the hero loader briefly switches image bitmaps off: parse other packs only after it
        if (!now && (!HERO_GLTF || (typeof window.createImageBitmap !== 'function' && 'ImageBitmap' in window))) { setTimeout(go, 40); return; }
        new THREE.GLTFLoader().parse(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength), '', resolve, reject);
      })();
    });
  }
  $('curtain-text').textContent = '유적지를 불러오는 중…';
  // everything downloads at once from the start: the village, the land, the castle and the monsters arrive
  // in the background while the ruins are being built and played
  setTimeout(function () { loadVillage(); loadLand(); loadCastleAndMonsters(); }, 0);
  Promise.all([fetchGz('forest.b64.txt'), fetchGz('forest_inst.b64.txt'), fetch('forest.json').then(function (r) { return r.json(); })]).then(function (res) {
    var inst = new Float32Array(res[1].buffer, res[1].byteOffset, res[1].byteLength / 4);
    return parseBytes(res[0], true).then(function (gltf) { setupWorld(gltf); buildForest(gltf, inst, res[2]); });
  }).then(function () {
    parseGlb('hero-glb', false, function (heroGltf) {
      setupHero(heroGltf);
      HERO_GLTF = heroGltf;
      curtain.hidden = true;
      devMenu.hidden = false;
      $('start-intro').focus();
      loadNpcPack();
      // Handy from the console: inspect the state, or advance the game by hand.
      window.__game = { hero: hero, cam: cam, talk: talk, tick: update, advance: advance, found: found, points: function () { return POINTS; },
                        groundAt: groundAt, flags: flags, render: function () { renderer.render(scene, camera); },
                        summon: function () { foundCount = 3; found[0] = found[1] = found[2] = true; finished = true; summon(); },
                        village: function () { return VILLAGE; }, allFound: function () { foundCount = 3; found[0] = found[1] = found[2] = true; }, zone: function () { return zone; }, interact: interact,
                        scene: scene, camera: camera, begin: begin, monsters: function () { return MONSTERS; }, combat: COMBAT, land: LAND, boxes: function () { return BOXES; },
    pick: PICK, res: RES, inv: INV, quest: QUEST, spawnPickup: spawnPickup, useItem: useItem, toggleBag: toggleBag, hold: hold, npcs: function () { return NPCS; }, forest: function () { return FOREST; } };
      frame();
    });
  }).catch(function (err) { console.error(err); fail('유적지를 불러오지 못했습니다: ' + (err && err.message ? err.message : err)); });
})();

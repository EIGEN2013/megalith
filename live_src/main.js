
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
    if (zone === 'village') return VILLAGE.groundAt(x, z);
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
      var l = toLocal(b, p.x, p.z), ex = b.hx + RADIUS, ez = b.hz + RADIUS;
      if (Math.abs(l[0]) >= ex || Math.abs(l[1]) >= ez) continue;
      var px = ex - Math.abs(l[0]), pz = ez - Math.abs(l[1]);
      if (px < pz) l[0] = l[0] < 0 ? -ex : ex; else l[1] = l[1] < 0 ? -ez : ez;
      p.x = b.x + l[0] * b.cos + l[1] * b.sin;
      p.z = b.z - l[0] * b.sin + l[1] * b.cos;
      v.x *= 0.6; v.z *= 0.6;
    }
  }

  // ---------------------------------------------------------------- renderer + sky + light
  var canvas = $('view'), stage = $('stage');
  var renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  function srgb(hex) { return new THREE.Color(hex).convertSRGBToLinear(); }   // authored in sRGB, lit in linear

  var scene = new THREE.Scene();
  scene.background = srgb(0xc4d6dc);
  scene.fog = new THREE.Fog(srgb(0xc4d6dc), 22, 150);
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

  var hemi = new THREE.HemisphereLight(srgb(0xdfeaff), srgb(0x6f7a5c), 0.85);
  scene.add(hemi);
  var sun = new THREE.DirectionalLight(srgb(0xfff0d2), 1.15);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.left = -16; sun.shadow.camera.right = 16;
  sun.shadow.camera.top = 16; sun.shadow.camera.bottom = -16;
  sun.shadow.camera.near = 1; sun.shadow.camera.far = 70;
  sun.shadow.bias = -0.0012; sun.shadow.normalBias = 0.03;
  scene.add(sun, sun.target);
  var SUN_OFFSET = new THREE.Vector3(10, 22, 14);

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

  var cam = { yaw: 0, pitch: 0.42, distance: 3.4, goal: 3.4 };
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

  var stickVec = { x: 0, y: 0 }, runLatch = false;
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
  $('btn-run').addEventListener('click', function () {
    runLatch = !runLatch;
    this.classList.toggle('on', runLatch);
    this.setAttribute('aria-pressed', String(runLatch));
  });
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
        if (m.map) m.map.anisotropy = renderer.capabilities.getMaxAnisotropy();   // keep the grass crisp at a distance
      } else if (/^(trunks|trilithon|door_stone|altar|monolith|rune_stone|small_stone|rock_|rune_pillar)/.test(n)) {
        o.castShadow = true; o.receiveShadow = true;
      } else if (n === 'leaves') {
        o.castShadow = true;   // shadows follow the leaf shapes, not the cards they are painted on
        o.customDepthMaterial = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking, map: m.map, alphaTest: 0.45 });
      }
      if (n === 'moon') { m.fog = false; m.emissiveIntensity = 1.0; }
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

  function setupHero(gltf) {
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
      if (['Jump', 'DoubleJump', 'Attack', 'WakeUp'].indexOf(clip.name) >= 0) {
        action.setLoop(THREE.LoopOnce, 1);
        action.clampWhenFinished = true;
      }
      hero.actions[clip.name] = action;
    });
    hero.mixer.addEventListener('finished', function (e) {
      var name = e.action.getClip().name;
      if (name === 'Attack') hero.attacking = false;
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
    if (zone === 'village') { $('objective').textContent = STORY.village.objective; return; }
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
    } else if (pt.id === 'compass') {
      say(entry.lines);
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
    }
    var el = $('prompt');
    if (!nearest) { if (!el.hidden) el.hidden = true; return; }
    var name = STORY.interact[nearest.id].name;
    if (el.hidden || $('prompt-name').textContent !== name) { $('prompt-name').textContent = name; el.hidden = false; }
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
    var running = keys.run || runLatch || Math.hypot(stickVec.x, stickVec.y) > 0.92;
    var speed = running ? RUN_SPEED : WALK_SPEED;
    if (hero.attacking && hero.onFloor) speed = 0;
    var accel = (hero.onFloor ? ACCEL : AIR_ACCEL) * dt;
    v.x = approach(v.x, dirX * speed, accel);
    v.z = approach(v.z, dirZ * speed, accel);

    if (free) {
      if (pressed.interact && nearest) interact(nearest);
      else if (pressed.jump) {
        if (hero.onFloor) { hero.jumpsUsed = 1; v.y = JUMP_VELOCITY; hero.onFloor = false; hero.attacking = false; replay('Jump', 0.08); }
        else if (hero.jumpsUsed < MAX_JUMPS) { hero.jumpsUsed = MAX_JUMPS; v.y = DOUBLE_JUMP_VELOCITY; hero.flipping = true; replay('DoubleJump', 0.05); }
      } else if (pressed.attack && hero.onFloor && !hero.attacking) { hero.attacking = true; replay('Attack', BLEND, ATTACK_SPEED); }
    }
    pressed.jump = pressed.attack = pressed.interact = false;

    if ((dirX || dirZ) && !(hero.attacking && hero.onFloor)) {
      var want = Math.atan2(dirX, dirZ), diff = Math.atan2(Math.sin(want - hero.heading), Math.cos(want - hero.heading));
      hero.heading += diff * Math.min(1, TURN_SPEED * dt);
    }

    // Sideways first (stones, trunks, the world's edge), then vertical (ground and stone tops).
    p.x += v.x * dt; p.z += v.z * dt;
    pushOut(p, v);
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
    var lying = hero.cutscene && !intro.woke;
    want.set(hero.pos.x, hero.pos.y + (lying ? 0.45 : 1.3), hero.pos.z);
    focus.lerp(want, Math.min(1, (lying ? 3 : 8) * dt));
    var cp = Math.cos(cam.pitch);
    var cx = focus.x + Math.sin(cam.yaw) * cp * cam.distance, cz = focus.z + Math.cos(cam.yaw) * cp * cam.distance;
    camera.position.set(cx, Math.max(groundAt(cx, cz) + 0.35, focus.y + Math.sin(cam.pitch) * cam.distance), cz);
    camera.lookAt(focus);
    if (doorFx.shake && zone === 'forest') { camera.position.x += (Math.random() - 0.5) * doorFx.shake; camera.position.y += (Math.random() - 0.5) * doorFx.shake; }
    sky.position.copy(camera.position);
    sun.target.position.copy(focus);
    sun.position.copy(focus).add(SUN_OFFSET);
  }

  var clock = new THREE.Clock();
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
  function buildVillage(gltf, inst, data) {
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
    });
    drop.forEach(function (o) { o.parent.remove(o); });
    var mtx = new THREE.Matrix4(), col = new THREE.Color();
    data.instances.forEach(function (gi) {
      var pm = protos[gi.mesh];
      if (!pm) return;
      var geo = bakedGeometry(pm), mat = pm.material;
      mat.vertexColors = true; mat.metalness = 0;
      if (gi.set === 'grass') mat.side = THREE.DoubleSide;
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
    V.ready = true;
    VILLAGE = V;
    // compile the village shaders now, while nobody is looking, so the summon does not stutter
    group.visible = true; renderer.compile(scene, camera); group.visible = false;
    var w = villageWait; villageWait = [];
    w.forEach(function (fn) { fn(); });
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
    Promise.all([get('village.b64.txt'), get('village_inst.b64.txt'), fetch('village.json').then(function (r) { return r.json(); })]).then(function (res) {
      var glb = fflate.gunzipSync(new Uint8Array(res[0]));
      var instBytes = fflate.gunzipSync(new Uint8Array(res[1]));
      var inst = new Float32Array(instBytes.buffer, instBytes.byteOffset, instBytes.byteLength / 4);
      new THREE.GLTFLoader().parse(glb.buffer.slice(glb.byteOffset, glb.byteOffset + glb.byteLength), '', function (g) { buildVillage(g, inst, res[2]); },
        function (err) { console.error('village', err); });
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
  function enterVillage() {
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
    var fl = $('flash');
    $('flash-text').textContent = '';
    setTimeout(function () { fl.style.transition = 'opacity 2.8s ease-out'; fl.classList.remove('on'); }, 250);
    setTimeout(function () {
      say(STORY.village.arrive, function () { setObjective(); stage.classList.remove('talking'); });
    }, 1900);
  }

  parseGlb('world-glb', false, function (worldGltf) {
    setupWorld(worldGltf);
    parseGlb('hero-glb', false, function (heroGltf) {
      setupHero(heroGltf);
      curtain.hidden = true;
      startIntro();
      // Handy from the console: inspect the state, or advance the game by hand.
      loadVillage();
      window.__game = { hero: hero, cam: cam, talk: talk, tick: update, advance: advance, found: found, points: function () { return POINTS; },
                        groundAt: groundAt, flags: flags, render: function () { renderer.render(scene, camera); },
                        summon: function () { foundCount = 3; found[0] = found[1] = found[2] = true; finished = true; summon(); },
                        village: function () { return VILLAGE; }, allFound: function () { foundCount = 3; found[0] = found[1] = found[2] = true; }, zone: function () { return zone; }, interact: interact,
                        scene: scene, camera: camera };
      frame();
    });
  });
})();

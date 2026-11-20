extends Node3D
## Megalith: the opening, the dialogue box, things to examine and the world's glowing parts.
##
## The words live in story.json and the places in world.json (written by tools/build_world.py),
## so the story can be edited without touching this script.

const STORY_PATH := "res://story.json"
const WORLD_PATH := "res://world.json"
const TYPE_SPEED := 42.0           # characters per second
const AMBER := Color(1.0, 0.63, 0.25)
const INK := Color(0.953, 0.937, 0.894)
const INK_DIM := Color(0.77, 0.757, 0.698)
const MOSS := Color(0.50, 0.68, 0.40)
const RUNE := Color(0.56, 0.90, 0.97)

@onready var player: CharacterBody3D = $Player
@onready var camera_rig: Node3D = $CameraRig
@onready var world: Node3D = $World
@onready var sun: DirectionalLight3D = $Sun

var story: Dictionary
var world_data: Dictionary

# Dialogue
var talking := false
var _lines: Array = []
var _index := 0
var _shown := 0.0
var _on_done := Callable()

# Story state
var found: Array[bool] = [false, false, false]
var found_count := 0
var finished := false
var intro_done := false
var _woke := false
var _intro_ending := false
var _left_circle := false
var _edge_cooldown := 0.0
var _nearest: Dictionary = {}

# World effects
var _time := 0.0
var _pulse := 0.0
var _rune_materials: Array[StandardMaterial3D] = []
var _memory_materials: Array = [null, null, null]
var _portal_material: StandardMaterial3D
var _beacons: Array = [null, null, null]

# UI
var _veil: ColorRect
var _nameplate: PanelContainer
var _objective: Label
var _legend: PanelContainer
var _prompt: PanelContainer
var _prompt_label: Label
var _dialogue: PanelContainer
var _who: Label
var _text: Label
var _hint: Label


func _ready() -> void:
	story = JSON.parse_string(FileAccess.get_file_as_string(STORY_PATH))
	world_data = JSON.parse_string(FileAccess.get_file_as_string(WORLD_PATH))
	sun.look_at_from_position(Vector3(10, 22, 14), Vector3.ZERO)
	_setup_world()
	_build_ui()

	# Robin starts asleep in the middle of the circle, under a black screen.
	var spawn: Dictionary = world_data["spawn"]
	player.global_position = Vector3(spawn["x"], 3.3, spawn["z"])
	player.get_node("Model").rotation.y = spawn["heading"]
	player.start_cutscene("Sleep")
	camera_rig.distance = 3.4
	camera_rig.goal_distance = 3.4
	camera_rig.look_height = 0.45
	camera_rig.goal_look_height = 0.45
	say(story["intro"], _on_intro_spoken)


func _process(delta: float) -> void:
	_time += delta
	_edge_cooldown = maxf(0.0, _edge_cooldown - delta)
	_update_dialogue(delta)
	_update_effects(delta)
	if _intro_ending and not player.is_clip_playing("WakeUp"):
		_finish_intro()
	if intro_done and not talking:
		_update_exploration()


func _unhandled_input(event: InputEvent) -> void:
	if talking:
		if event.is_action_pressed("advance"):
			_advance()
			get_viewport().set_input_as_handled()
	elif event.is_action_pressed("interact") and not _nearest.is_empty() and not player.controls_locked:
		_interact(_nearest)
		get_viewport().set_input_as_handled()


# ------------------------------------------------------------------ dialogue
## Shows the lines one by one; on_done runs after the last one is dismissed.
func say(lines: Array, on_done := Callable()) -> void:
	if lines.is_empty():
		if on_done.is_valid():
			on_done.call()
		return
	_lines = lines
	_index = 0
	_on_done = on_done
	talking = true
	player.controls_locked = true
	player.unlock_after(0.0)
	_dialogue.visible = true
	_prompt.visible = false
	_legend.visible = false
	_show_line()


func _show_line() -> void:
	var line: Dictionary = _lines[_index]
	var who: String = line["who"]
	_shown = 0.0
	_who.text = story["speakers"].get(who, "")
	_who.visible = _who.text != ""
	_who.add_theme_color_override("font_color", RUNE if who == "voice" else MOSS)
	_text.add_theme_color_override("font_color", RUNE if who == "voice" else (INK_DIM if who == "narration" else INK))
	_text.text = line["text"]
	_text.visible_characters = 0
	_hint.modulate.a = 0.0
	if line.has("stage"):
		_on_stage(line["stage"])


func _advance() -> void:
	var total := (_lines[_index]["text"] as String).length()
	if _shown < total:
		_shown = total          # first press finishes the line, the next one moves on
		_text.visible_characters = -1
		_hint.modulate.a = 1.0
		return
	_index += 1
	if _index < _lines.size():
		_show_line()
		return
	talking = false
	_dialogue.visible = false
	_legend.visible = intro_done
	if not player.cutscene:
		player.unlock_after(0.2)
	var done := _on_done
	_on_done = Callable()
	if done.is_valid():
		done.call()


func _update_dialogue(delta: float) -> void:
	if not talking:
		return
	var total := (_lines[_index]["text"] as String).length()
	if _shown >= total:
		return
	_shown = minf(total, _shown + delta * TYPE_SPEED)
	_text.visible_characters = int(_shown)
	if _shown >= total:
		_text.visible_characters = -1
		_hint.modulate.a = 1.0


# ------------------------------------------------------------------ story
func _on_stage(stage: String) -> void:
	match stage:
		"open_eyes":
			create_tween().tween_property(_veil, "color:a", 0.0, 2.8).set_ease(Tween.EASE_OUT)
		"wake", "standing":
			if not _woke:
				_woke = true
				player.start_cutscene("WakeUp", 0.5)
				camera_rig.goal_distance = 4.4
				camera_rig.goal_look_height = 1.3
		"pulse":
			_pulse = 2.8


func _on_intro_spoken() -> void:
	_intro_ending = true   # control returns once Robin is on his feet


func _finish_intro() -> void:
	_intro_ending = false
	intro_done = true
	player.end_cutscene()
	player.unlock_after(0.1)
	camera_rig.goal_distance = 5.6
	_update_objective()
	_nameplate.visible = true
	_legend.visible = true


func _update_objective() -> void:
	var text: Dictionary = story["objective"]
	if finished:
		_objective.text = "메갈리스의 문이 열리려 한다"
	elif found_count == 0:
		_objective.text = text["start"]
	elif found_count < 3:
		_objective.text = (text["progress"] as String).replace("{n}", str(found_count))
	else:
		_objective.text = text["done"]


func _interact(point: Dictionary) -> void:
	var id: String = point["id"]
	var entry: Dictionary = story["interact"][id]
	player.velocity = Vector3.ZERO
	if id.begins_with("memory_"):
		var i := int(id.trim_prefix("memory_")) - 1
		if found[i]:
			say(entry["lines_again"])
			return
		_pulse = 1.5
		say(entry["lines"], _on_memory_recovered.bind(i))
	elif id == "door" and found_count == 3:
		_pulse = 3.0
		say(entry["lines_done"], _on_door_answered)
	else:
		say(entry["lines"])


func _on_memory_recovered(i: int) -> void:
	found[i] = true
	found_count += 1
	_update_objective()
	if found_count == 3:
		say(story["events"]["all_memories"])


func _on_door_answered() -> void:
	finished = true
	_update_objective()


func _update_exploration() -> void:
	var here := player.global_position
	var flat := Vector2(here.x, here.z)

	# The mist closes the valley in.
	var bounds: float = world_data["bounds"]
	if flat.length() > bounds:
		flat = flat.normalized() * bounds
		player.global_position = Vector3(flat.x, here.y, flat.y)
		if _edge_cooldown <= 0.0:
			_edge_cooldown = 20.0
			player.velocity = Vector3.ZERO
			say(story["events"]["world_edge"])
			return
	if not _left_circle and flat.length() > float(world_data["plateau_radius"]) + 7.0:
		_left_circle = true
		say(story["events"]["leave_circle"])
		return

	# What is close enough to examine?
	_nearest = {}
	if not player.controls_locked and player.is_on_floor():
		var best := INF
		for point: Dictionary in world_data["interact"]:
			var d := flat.distance_to(Vector2(point["x"], point["z"]))
			if d < float(point["radius"]) and d < best and absf(here.y - float(point["y"])) < 2.5:
				best = d
				_nearest = point
	_prompt.visible = not _nearest.is_empty()
	if _prompt.visible:
		_prompt_label.text = "F   %s 살펴보기" % story["interact"][_nearest["id"]]["name"]


# ------------------------------------------------------------------ world
func _setup_world() -> void:
	for node in world.find_children("*", "MeshInstance3D", true, false):
		var instance := node as MeshInstance3D
		var material := instance.mesh.surface_get_material(0) as StandardMaterial3D
		if material == null:
			continue
		var material_name := material.resource_name
		# The world is painted per vertex (linear colours); the importer leaves that switched off.
		material.vertex_color_use_as_albedo = true
		if material_name == "moon":
			material.disable_fog = true
			instance.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		elif material_name == "far" or material_name == "leaf":
			instance.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		elif material_name.begins_with("glow_"):
			material.emission_energy_multiplier = 0.8   # the bloom does the rest
		elif material_name.begins_with("rune_"):
			_rune_materials.append(material)
		elif material_name == "portal":
			_portal_material = material
		elif material_name.begins_with("memory_"):
			_memory_materials[int(material_name.trim_prefix("memory_")) - 1] = material

	# The modelled stones and rocks collide as simple boxes (the same ones the web build uses).
	var solids := StaticBody3D.new()
	add_child(solids)
	for collider: Dictionary in world_data["colliders"]:
		if not collider.has("a"):
			continue
		var shape := BoxShape3D.new()
		shape.size = Vector3(float(collider["hx"]) * 2.0, float(collider["y1"]) - float(collider["y0"]), float(collider["hz"]) * 2.0)
		var solid := CollisionShape3D.new()
		solid.shape = shape
		solid.position = Vector3(collider["x"], (float(collider["y0"]) + float(collider["y1"])) * 0.5, collider["z"])
		solid.rotation.y = collider["yaw"]
		solids.add_child(solid)

	# Each memory stone gets a warm light and a faint beacon, so it can be found from afar.
	for point: Dictionary in world_data["interact"]:
		var id: String = point["id"]
		if not id.begins_with("memory_"):
			continue
		var at := Vector3(point["x"], point["y"], point["z"])
		var light := OmniLight3D.new()
		light.light_color = AMBER
		light.omni_range = 16.0
		light.light_energy = 2.0
		light.position = at + Vector3.UP * 2.2
		add_child(light)
		var shaft := CylinderMesh.new()
		shaft.top_radius = 0.16
		shaft.bottom_radius = 0.34
		shaft.height = 46.0
		shaft.cap_top = false
		shaft.cap_bottom = false
		var glow := StandardMaterial3D.new()
		glow.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
		glow.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
		glow.blend_mode = BaseMaterial3D.BLEND_MODE_ADD
		glow.cull_mode = BaseMaterial3D.CULL_DISABLED
		glow.disable_fog = true
		glow.albedo_color = Color(AMBER, 0.3)
		var beam := MeshInstance3D.new()
		beam.mesh = shaft
		beam.material_override = glow
		beam.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		beam.position = at + Vector3.UP * 25.0
		add_child(beam)
		_beacons[int(id.trim_prefix("memory_")) - 1] = {"light": light, "glow": glow}


func _update_effects(delta: float) -> void:
	_pulse = maxf(0.0, _pulse - delta)
	var breathe := 0.5 + 0.5 * sin(_time * 1.7)
	var boost := 0.0
	if _pulse > 0.0:
		boost = 1.6 * sin(minf(1.0, _pulse) * PI * 0.5) * (0.7 + 0.3 * sin(_time * 9.0))
	for material in _rune_materials:
		material.emission_energy_multiplier = 0.5 + 0.2 * breathe + boost
	for i in 3:
		var memory := _memory_materials[i] as StandardMaterial3D
		if memory:
			memory.emission_energy_multiplier = 1.4 if found[i] else 0.35 + 0.8 * (0.5 + 0.5 * sin(_time * 2.6 + i * 2.0))
		if _beacons[i]:
			(_beacons[i]["glow"] as StandardMaterial3D).albedo_color.a = 0.07 if found[i] else 0.2 + 0.12 * breathe
			(_beacons[i]["light"] as OmniLight3D).light_energy = 1.2 if found[i] else 1.4 + 1.2 * breathe
	if _portal_material:
		var open := found_count == 3
		_portal_material.emission_energy_multiplier = (1.1 if open else 0.6) + (0.35 if open else 0.12) * sin(_time * 2.2) + boost * 0.4


# ------------------------------------------------------------------ ui
func _build_ui() -> void:
	var layer := CanvasLayer.new()
	add_child(layer)
	var root := Control.new()
	root.set_anchors_preset(Control.PRESET_FULL_RECT)
	root.mouse_filter = Control.MOUSE_FILTER_IGNORE
	# Hangul needs a font that has it; take one from the system.
	var font := SystemFont.new()
	font.font_names = PackedStringArray(["Malgun Gothic", "Apple SD Gothic Neo", "Noto Sans CJK KR", "Noto Sans KR", "NanumGothic", "sans-serif"])
	var theme := Theme.new()
	theme.default_font = font
	theme.default_font_size = 16
	root.theme = theme
	layer.add_child(root)

	_veil = ColorRect.new()
	_veil.color = Color(0.043, 0.055, 0.047, 1.0)
	_veil.set_anchors_preset(Control.PRESET_FULL_RECT)
	_veil.mouse_filter = Control.MOUSE_FILTER_IGNORE
	root.add_child(_veil)

	# Title and current goal, top-left.
	_nameplate = _panel(root)
	_nameplate.position = Vector2(16, 14)
	_nameplate.visible = false
	var plate := _column(_nameplate, 4)
	_label(plate, story["title"], 26, INK)
	_objective = _label(plate, "", 15, INK_DIM)

	# Controls, bottom-left.
	_legend = _panel(root)
	_pin_bottom(_legend, 0.0, 16.0, 16.0)
	_legend.visible = false
	_label(_legend, "WASD  걷기      Shift  달리기\nSpace  점프 (공중에서 한 번 더: 2단 점프)\nJ / 클릭  공격      F  살펴보기\nQ · E / 우클릭 드래그  카메라      휠  확대", 14, INK_DIM)

	# "Examine" prompt and the dialogue box, bottom-centre.
	_prompt = _panel(root)
	_pin_bottom(_prompt, 0.5, 0.0, 26.0)
	_prompt.grow_horizontal = Control.GROW_DIRECTION_BOTH
	_prompt.visible = false
	_prompt_label = _label(_prompt, "", 16, INK)

	_dialogue = _panel(root)
	_pin_bottom(_dialogue, 0.5, -380.0, 22.0)
	_dialogue.offset_right = 380.0
	_dialogue.visible = false
	var box := _column(_dialogue, 6)
	_who = _label(box, "", 15, MOSS)
	_text = _label(box, "", 22, INK)
	_text.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_text.custom_minimum_size = Vector2(0, 68)
	_hint = _label(box, "Space · 클릭 ▸", 13, INK_DIM)
	_hint.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT


func _panel(parent: Control) -> PanelContainer:
	var panel := PanelContainer.new()
	var style := StyleBoxFlat.new()
	style.bg_color = Color(0.086, 0.106, 0.086, 0.8)
	style.border_color = Color(0.95, 0.94, 0.89, 0.18)
	style.set_border_width_all(1)
	style.set_corner_radius_all(3)
	style.content_margin_left = 16
	style.content_margin_right = 16
	style.content_margin_top = 11
	style.content_margin_bottom = 11
	panel.add_theme_stylebox_override("panel", style)
	panel.mouse_filter = Control.MOUSE_FILTER_IGNORE
	parent.add_child(panel)
	return panel


func _column(parent: Control, gap: int) -> VBoxContainer:
	var column := VBoxContainer.new()
	column.add_theme_constant_override("separation", gap)
	column.mouse_filter = Control.MOUSE_FILTER_IGNORE
	parent.add_child(column)
	return column


func _label(parent: Control, text: String, size: int, colour: Color) -> Label:
	var label := Label.new()
	label.text = text
	label.add_theme_font_size_override("font_size", size)
	label.add_theme_color_override("font_color", colour)
	parent.add_child(label)
	return label


## Anchors a panel to the bottom edge so it grows upward as its content grows.
func _pin_bottom(panel: Control, anchor_x: float, left: float, margin_bottom: float) -> void:
	panel.anchor_left = anchor_x
	panel.anchor_right = anchor_x
	panel.anchor_top = 1.0
	panel.anchor_bottom = 1.0
	panel.offset_left = left
	panel.offset_right = left
	panel.offset_top = -margin_bottom
	panel.offset_bottom = -margin_bottom
	panel.grow_vertical = Control.GROW_DIRECTION_BEGIN

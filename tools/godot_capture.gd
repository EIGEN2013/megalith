extends SceneTree
## Smoke test: plays the opening, examines a memory stone, runs into the forest, then jumps and
## attacks, saving a screenshot at each step.
## Usage: godot --path hero_game -s ../tools/godot_capture.gd -- <output dir>

var out_dir := "user://"
var log_lines: PackedStringArray = []
var main: Node3D
var player: CharacterBody3D
var anim: AnimationPlayer
var rig: Node3D


func _initialize() -> void:
	var args := OS.get_cmdline_user_args()
	if args.size() > 0:
		out_dir = args[0]
	_run()


func _run() -> void:
	main = (load("res://main.tscn") as PackedScene).instantiate()
	root.add_child(main)
	player = main.get_node("Player")
	anim = player.get_node("Model/AnimationPlayer")
	rig = main.get_node("CameraRig")
	_note("clips: %s" % ", ".join(anim.get_animation_list()))

	await _wait(1.2)
	await _shot("1_dark_first_line")
	await _tap("advance"); await _tap("advance")      # finish line 1, show line 2 (eyes open)
	await _wait(3.2)
	await _shot("2_asleep")
	while main.talking and main._index < 3:
		await _tap("advance")
	await _wait(2.4)
	await _shot("3_getting_up")
	await _skip_dialogue()
	var waited := 0.0
	while not main.intro_done and waited < 10.0:
		await _wait(0.25)
		waited += 0.25
	await _wait(0.5)
	await _shot("4_awake")
	_note("intro_done=%s locked=%s objective=%s" % [main.intro_done, player.controls_locked, main._objective.text])

	var stone: Dictionary = main.world_data["interact"][2]
	await _walk_to(Vector2(stone["x"], stone["z"]), 1.4, 10.0)
	await _wait(0.4)
	await _shot("5_at_memory_stone")
	_note("prompt visible=%s text=%s" % [main._prompt.visible, main._prompt_label.text])
	await _tap("interact")
	await _wait(1.0)
	await _shot("6_memory_line")
	await _skip_dialogue()
	_note("found=%s objective=%s" % [main.found, main._objective.text])

	rig.yaw = PI
	Input.action_press("run"); Input.action_press("move_forward")
	await _wait(5.0)
	Input.action_release("move_forward"); Input.action_release("run")
	rig.yaw = 0.5
	await _wait(0.6)
	await _shot("7_forest_line")
	_note("talking=%s left_circle=%s" % [main.talking, main._left_circle])
	await _skip_dialogue()
	await _wait(0.5)

	await _tap("jump")
	await _wait(0.25)
	await _tap("jump")
	await _wait(0.25)
	await _shot("8_double_jump")
	await _wait(1.2)
	await _tap("attack")
	await _wait(0.5)
	await _shot("9_attack")
	await _wait(1.6)
	_note("end: anim=%s on_floor=%s pos=%s" % [anim.current_animation, player.is_on_floor(), player.global_position.snapped(Vector3.ONE * 0.01)])

	var f := FileAccess.open(out_dir.path_join("capture_log.txt"), FileAccess.WRITE)
	f.store_string("\n".join(log_lines))
	f.close()
	quit()


func _wait(seconds: float) -> void:
	await create_timer(seconds).timeout


func _tap(action: String) -> void:
	for down in [true, false]:
		var event := InputEventAction.new()
		event.action = action
		event.pressed = down
		Input.parse_input_event(event)
		await _wait(0.06)


func _skip_dialogue() -> void:
	var guard := 0
	while main.talking and guard < 60:
		await _tap("advance")
		guard += 1


func _walk_to(target: Vector2, stop_at: float, limit: float) -> void:
	var spent := 0.0
	Input.action_press("move_forward")
	while spent < limit:
		var here := Vector2(player.global_position.x, player.global_position.z)
		if here.distance_to(target) < stop_at:
			break
		rig.yaw = atan2(-(target.x - here.x), -(target.y - here.y))
		await physics_frame
		spent += 1.0 / 60.0
	Input.action_release("move_forward")


func _shot(name: String) -> void:
	await RenderingServer.frame_post_draw
	root.get_texture().get_image().save_png(out_dir.path_join(name + ".png"))
	_note("%s: anim=%s pos=%s talking=%s on_floor=%s" % [name, anim.current_animation, player.global_position.snapped(Vector3.ONE * 0.01), main.talking, player.is_on_floor()])


func _note(text: String) -> void:
	print(text)
	log_lines.append(text)

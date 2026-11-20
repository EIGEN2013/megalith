extends CharacterBody3D
## Robin: walk, run, jump, double jump (front flip) and attack, plus cutscene clips.
##
## Controls (registered in code below, so the project needs no Input Map setup):
##   WASD / arrows  move          Shift        run
##   Space          jump, again in the air = double jump
##   J / left click attack        F            examine what is nearby
##   Space / Enter / F / click    next line while someone is talking

@export_group("Movement")
@export var walk_speed := 1.9
@export var run_speed := 5.2
@export var acceleration := 28.0
@export var air_acceleration := 10.0
@export var turn_speed := 12.0

@export_group("Jump")
@export var gravity := 20.0
@export var jump_velocity := 7.0
@export var double_jump_velocity := 6.6
@export var max_jumps := 2

@export_group("Animation")
## Ground speed (m/s) at which the Walk / Run clips look right at 1x playback.
@export var walk_clip_speed := 1.6
@export var run_clip_speed := 4.2
@export var attack_playback_speed := 1.5
@export var blend_time := 0.15

const LOOPING := ["Idle", "Walk", "Run", "Fall", "Sleep"]

@onready var model: Node3D = $Model
@onready var anim: AnimationPlayer = $Model/AnimationPlayer

## While true the player ignores movement, jump and attack input (dialogue, cutscenes).
var controls_locked := false
## While true the controller leaves the animation alone (a cutscene clip is playing).
var cutscene := false
var jumps_used := 0
var attacking := false
var flipping := false
var _unlock_timer := 0.0


func _enter_tree() -> void:
	_register_inputs()


func _ready() -> void:
	floor_snap_length = 0.4   # stay glued to the ground on the way down the mound
	for clip in LOOPING:
		anim.get_animation(clip).loop_mode = Animation.LOOP_LINEAR
	anim.animation_finished.connect(_on_animation_finished)
	anim.play("Idle")


## Plays a clip the controller will not override until end_cutscene().
func start_cutscene(clip: String, blend := 0.0) -> void:
	cutscene = true
	controls_locked = true
	velocity = Vector3.ZERO
	anim.speed_scale = 1.0
	anim.play(clip, blend)


func end_cutscene() -> void:
	cutscene = false
	anim.play("Idle", 0.4)


func is_clip_playing(clip: String) -> bool:
	return anim.is_playing() and anim.current_animation == clip


## Hands control back after a short beat, so the key that closed a dialogue does not also jump.
func unlock_after(delay := 0.2) -> void:
	_unlock_timer = delay


func _physics_process(delta: float) -> void:
	if _unlock_timer > 0.0:
		_unlock_timer -= delta
		if _unlock_timer <= 0.0:
			controls_locked = false

	var on_floor := is_on_floor()
	if on_floor:
		jumps_used = 0
		flipping = false
	else:
		velocity.y -= gravity * delta

	# Movement is relative to where the camera looks.
	var input := Vector2.ZERO if controls_locked else Input.get_vector("move_left", "move_right", "move_forward", "move_back")
	var direction := Vector3.ZERO
	var camera := get_viewport().get_camera_3d()
	if camera and input.length() > 0.05:
		var forward := -camera.global_transform.basis.z
		var right := camera.global_transform.basis.x
		forward.y = 0.0
		right.y = 0.0
		direction = (right.normalized() * input.x - forward.normalized() * input.y).normalized()

	var speed := run_speed if Input.is_action_pressed("run") else walk_speed
	if attacking and on_floor:
		speed = 0.0  # plant the feet while punching
	var accel := acceleration if on_floor else air_acceleration
	var target := direction * speed
	velocity.x = move_toward(velocity.x, target.x, accel * delta)
	velocity.z = move_toward(velocity.z, target.z, accel * delta)

	if not controls_locked:
		if Input.is_action_just_pressed("jump"):
			_try_jump(on_floor)
		if Input.is_action_just_pressed("attack") and on_floor and not attacking:
			attacking = true
			anim.speed_scale = 1.0
			anim.play("Attack", blend_time, attack_playback_speed)

	if direction != Vector3.ZERO and not (attacking and on_floor):
		# The model faces +Z, so its yaw is the heading of the move direction.
		model.rotation.y = lerp_angle(model.rotation.y, atan2(direction.x, direction.z), turn_speed * delta)

	move_and_slide()
	if not cutscene:
		_update_animation()


func _try_jump(on_floor: bool) -> void:
	anim.speed_scale = 1.0
	if on_floor:
		jumps_used = 1
		velocity.y = jump_velocity
		attacking = false
		anim.play("Jump", 0.08)
	elif jumps_used < max_jumps:
		# Walking off a ledge spends the ground jump, leaving the air jump.
		jumps_used = max_jumps
		velocity.y = double_jump_velocity
		flipping = true
		anim.play("DoubleJump", 0.05)
		anim.seek(0.0, true)


func _update_animation() -> void:
	if attacking or flipping:
		return
	var ground_speed := Vector2(velocity.x, velocity.z).length()
	if not is_on_floor():
		if velocity.y <= 0.5 or anim.current_animation != "Jump":
			_play("Fall")
	elif ground_speed > (walk_speed + run_speed) * 0.5:
		_play("Run", clampf(ground_speed / run_clip_speed, 0.7, 1.6))
	elif ground_speed > 0.2:
		_play("Walk", clampf(ground_speed / walk_clip_speed, 0.6, 1.6))
	else:
		_play("Idle")


func _play(clip: String, playback_speed := 1.0) -> void:
	if anim.current_animation != clip:
		anim.play(clip, blend_time)
	anim.speed_scale = playback_speed


func _on_animation_finished(clip: StringName) -> void:
	if clip == &"Attack":
		attacking = false
	elif clip == &"DoubleJump":
		flipping = false
	anim.speed_scale = 1.0


func _register_inputs() -> void:
	var keys := {
		"move_forward": [KEY_W, KEY_UP],
		"move_back": [KEY_S, KEY_DOWN],
		"move_left": [KEY_A, KEY_LEFT],
		"move_right": [KEY_D, KEY_RIGHT],
		"run": [KEY_SHIFT],
		"jump": [KEY_SPACE],
		"attack": [KEY_J],
		"interact": [KEY_F],
		"advance": [KEY_SPACE, KEY_ENTER, KEY_KP_ENTER, KEY_F],
		"camera_left": [KEY_Q],
		"camera_right": [KEY_E],
	}
	for action in keys:
		if InputMap.has_action(action):
			continue
		InputMap.add_action(action)
		for keycode in keys[action]:
			var event := InputEventKey.new()
			event.physical_keycode = keycode
			InputMap.action_add_event(action, event)
	for action in ["attack", "advance"]:
		var click := InputEventMouseButton.new()
		click.button_index = MOUSE_BUTTON_LEFT
		if not InputMap.action_has_event(action, click):
			InputMap.action_add_event(action, click)

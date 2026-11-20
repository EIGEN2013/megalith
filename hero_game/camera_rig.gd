extends Node3D
## Orbit camera that follows the player and pulls in when stones or the ground get in the way.
## Q / E or right-mouse drag to look around, mouse wheel to zoom.

@export var target_path: NodePath
@export var look_height := 1.3
@export var distance := 5.6
@export var min_distance := 2.5
@export var max_distance := 10.0
@export var key_turn_speed := 2.2
@export var mouse_sensitivity := 0.005
@export var follow_speed := 8.0

@onready var camera: Camera3D = $Camera3D
@onready var target: CollisionObject3D = get_node(target_path)

var yaw := 0.0
var pitch := -0.42
## The distance and look height ease toward these, so cutscenes can move the camera gently.
var goal_distance := 5.6
var goal_look_height := 1.3


func _ready() -> void:
	goal_distance = distance
	goal_look_height = look_height
	global_position = target.global_position + Vector3.UP * look_height
	_apply(distance)


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseMotion and Input.is_mouse_button_pressed(MOUSE_BUTTON_RIGHT):
		yaw -= event.relative.x * mouse_sensitivity
		pitch = clampf(pitch - event.relative.y * mouse_sensitivity, -1.25, 0.25)
	elif event is InputEventMouseButton and event.pressed:
		if event.button_index == MOUSE_BUTTON_WHEEL_UP:
			goal_distance = maxf(min_distance, goal_distance - 0.6)
		elif event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			goal_distance = minf(max_distance, goal_distance + 0.6)


func _physics_process(delta: float) -> void:
	yaw += Input.get_axis("camera_right", "camera_left") * key_turn_speed * delta
	distance = lerpf(distance, goal_distance, clampf(2.5 * delta, 0.0, 1.0))
	look_height = lerpf(look_height, goal_look_height, clampf(3.0 * delta, 0.0, 1.0))
	var goal := target.global_position + Vector3.UP * look_height
	global_position = global_position.lerp(goal, clampf(follow_speed * delta, 0.0, 1.0))
	rotation = Vector3(pitch, yaw, 0.0)

	# Keep the view clear: stop the camera in front of whatever is between it and Robin.
	var wanted := global_position + global_transform.basis.z * distance
	var query := PhysicsRayQueryParameters3D.create(global_position, wanted)
	query.exclude = [target.get_rid()]
	var hit := get_world_3d().direct_space_state.intersect_ray(query)
	var clear := distance
	if hit:
		clear = clampf(global_position.distance_to(hit.position) - 0.3, 0.6, distance)
	_apply(clear)


func _apply(camera_distance: float) -> void:
	rotation = Vector3(pitch, yaw, 0.0)
	camera.position = Vector3(0.0, 0.0, camera_distance)

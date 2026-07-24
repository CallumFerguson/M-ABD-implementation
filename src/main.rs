use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::math::{DMat3, DQuat, DVec3};
use bevy::platform::time::Instant;
use bevy::prelude::*;
use bevy::text::FontSize;
use bevy::time::Fixed;
use bevy::window::{PresentMode, WindowResolution};

mod physx_demo;

use physx_demo::{PHYSX_AVAILABLE, PhysxDemo};

const DEFAULT_GRID_SIZE: usize = 10;
const GRID_SIZE_OPTIONS: [usize; 4] = [10, 25, 50, 100];
const GRID_SPACING: f64 = 0.55;
const DEFAULT_FIXED_HZ: f64 = 30.0;
const FIXED_HZ_OPTIONS: [f64; 5] = [30.0, 60.0, 120.0, 200.0, 500.0];
const GRAVITY: DVec3 = DVec3::new(0.0, -9.81, 0.0);
const VELOCITY_DAMPING_AT_DEFAULT_HZ: f64 = 0.997;
const AFFINE_STIFFNESS: f64 = 12_000.0;
const COROTATED_ITERATIONS: usize = 24;
const CONTACT_PASSES: usize = 2;
const DEMO_COUNT: usize = 3;

const HUB_RADIUS: f32 = 0.075;
const ROD_THICKNESS: f32 = 0.055;
const ROD_LENGTH_FACTOR: f32 = 0.78;
const BALL_RADIUS: f32 = 0.34;
const CYLINDER_RADIUS: f32 = 0.70;
const CYLINDER_LENGTH: f32 = 5.80;
const CONTACT_EPSILON: f64 = 1.0e-12;

const HUB_CENTER: [f64; 4] = [0.25, 0.25, 0.25, 0.25];
const ROD_START: [f64; 4] = [0.5, 0.5, 0.0, 0.0];
const ROD_END: [f64; 4] = [0.0, 0.0, 0.5, 0.5];

pub(crate) fn grid_span(grid_size: usize) -> f64 {
    grid_size.saturating_sub(1) as f64 * GRID_SPACING
}

pub(crate) fn cylinder_origin_z(grid_size: usize) -> f64 {
    let default_center_offset = -2.30 + grid_span(DEFAULT_GRID_SIZE) * 0.5;
    -grid_span(grid_size) * 0.5 + default_center_offset
}

pub(crate) fn cylinder_length(grid_size: usize) -> f64 {
    let default_end_margin = CYLINDER_LENGTH as f64 - grid_span(DEFAULT_GRID_SIZE);
    grid_span(grid_size) + default_end_margin
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum DemoScene {
    JointGrid,
    CylinderDrape,
    FallingBalls,
}

impl DemoScene {
    fn number(self) -> usize {
        match self {
            Self::JointGrid => 1,
            Self::CylinderDrape => 2,
            Self::FallingBalls => 3,
        }
    }

    fn title(self) -> &'static str {
        match self {
            Self::JointGrid => "Joint grid",
            Self::CylinderDrape => "Cylinder drape",
            Self::FallingBalls => "Falling balls",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SimulationBackend {
    Project,
    Physx,
}

impl SimulationBackend {
    fn toggled(self) -> Self {
        if !PHYSX_AVAILABLE {
            return Self::Project;
        }

        match self {
            Self::Project => Self::Physx,
            Self::Physx => Self::Project,
        }
    }
}

#[derive(Resource)]
struct ActiveDemo {
    scene: DemoScene,
    backend: SimulationBackend,
    grid_size: usize,
}

#[derive(Clone, Copy)]
enum BodyKind {
    Hub { fixed: bool },
    Rod,
    Ball,
}

#[derive(Clone)]
struct AffineBody {
    kind: BodyKind,
    fixed: bool,
    rest_points: [DVec3; 4],
    positions: [DVec3; 4],
    previous_positions: [DVec3; 4],
    predicted_positions: [DVec3; 4],
    velocities: [DVec3; 4],
    mass_per_point: f64,
    stiffness: f64,
    inertia: f64,
    inverse_diagonal: f64,
}

impl AffineBody {
    fn new(
        kind: BodyKind,
        rest_points: [DVec3; 4],
        translation: DVec3,
        rotation: DQuat,
        mass: f64,
        fixed: bool,
    ) -> Self {
        let positions = rest_points.map(|point| translation + rotation * point);

        Self {
            kind,
            fixed,
            rest_points,
            positions,
            previous_positions: positions,
            predicted_positions: positions,
            velocities: [DVec3::ZERO; 4],
            mass_per_point: mass / 4.0,
            stiffness: AFFINE_STIFFNESS,
            inertia: 0.0,
            inverse_diagonal: 0.0,
        }
    }

    fn centroid(&self) -> DVec3 {
        centroid(&self.positions)
    }

    fn rotation(&self) -> DMat3 {
        match self.kind {
            BodyKind::Rod => closest_rotation(rod_deformation_gradient(&self.positions)),
            BodyKind::Hub { .. } | BodyKind::Ball => DMat3::IDENTITY,
        }
    }

    fn predict(&mut self, dt: f64) {
        self.previous_positions = self.positions;

        if self.fixed {
            self.predicted_positions = self.positions;
            self.velocities = [DVec3::ZERO; 4];
            self.inverse_diagonal = 0.0;
            return;
        }

        self.inertia = self.mass_per_point / (dt * dt);
        self.inverse_diagonal = 1.0 / (self.inertia + self.stiffness);

        for index in 0..4 {
            self.predicted_positions[index] =
                self.positions[index] + self.velocities[index] * dt + GRAVITY * (dt * dt);
            self.positions[index] = self.predicted_positions[index];
        }
    }

    fn project_corotated_shape(&mut self) {
        if self.fixed {
            return;
        }

        let center = self.centroid();
        let rotation = match self.kind {
            BodyKind::Rod => closest_rotation(rod_deformation_gradient(&self.positions)),
            BodyKind::Hub { .. } | BodyKind::Ball => DMat3::IDENTITY,
        };

        for index in 0..4 {
            let rigid_target = center + rotation * self.rest_points[index];
            self.positions[index] = (self.predicted_positions[index] * self.inertia
                + rigid_target * self.stiffness)
                * self.inverse_diagonal;
        }
    }

    fn finish_step(&mut self, velocity_scale: f64) {
        if self.fixed {
            self.velocities = [DVec3::ZERO; 4];
            return;
        }

        for index in 0..4 {
            self.velocities[index] =
                (self.positions[index] - self.previous_positions[index]) * velocity_scale;
        }
    }
}

fn velocity_damping_for_dt(dt: f64) -> f64 {
    VELOCITY_DAMPING_AT_DEFAULT_HZ.powf(dt * DEFAULT_FIXED_HZ)
}

#[derive(Clone, Copy)]
struct Attachment {
    body: usize,
    weights: [f64; 4],
}

#[derive(Clone, Copy)]
struct BallJoint {
    a: Attachment,
    b: Attachment,
}

#[derive(Clone, Copy)]
struct CylinderCollider {
    origin: DVec3,
    axis: DVec3,
    radius: f64,
    length: f64,
}

#[derive(Resource)]
struct NetSimulation {
    scene: DemoScene,
    grid_size: usize,
    bodies: Vec<AffineBody>,
    joints: Vec<BallJoint>,
    cylinder: Option<CylinderCollider>,
    ball_indices: Vec<usize>,
    solver_scratch: SolverScratch,
}

struct SolverScratch {
    constraint_residual: Vec<DVec3>,
    solution: Vec<DVec3>,
    joint_rod_inverse_weight: Vec<f64>,
    hub_coupling: Vec<f64>,
    hub_inverse_rod_weight_sum: Vec<f64>,
    hub_weighted_residual: Vec<DVec3>,
    hub_schur_factor: Vec<f64>,
    contact_proxy_start: Vec<DVec3>,
    contact_proxy_end: Vec<DVec3>,
}

impl SolverScratch {
    fn new(body_count: usize, joint_count: usize) -> Self {
        Self {
            constraint_residual: vec![DVec3::ZERO; joint_count],
            solution: vec![DVec3::ZERO; joint_count],
            joint_rod_inverse_weight: vec![0.0; joint_count],
            hub_coupling: vec![0.0; body_count],
            hub_inverse_rod_weight_sum: vec![0.0; body_count],
            hub_weighted_residual: vec![DVec3::ZERO; body_count],
            hub_schur_factor: vec![0.0; body_count],
            contact_proxy_start: vec![DVec3::ZERO; body_count],
            contact_proxy_end: vec![DVec3::ZERO; body_count],
        }
    }
}

#[derive(Resource, Default)]
struct SimulationTiming {
    latest_step_ms: f64,
}

impl NetSimulation {
    fn new(scene: DemoScene) -> Self {
        Self::with_grid_size(scene, DEFAULT_GRID_SIZE)
    }

    fn with_grid_size(scene: DemoScene, grid_size: usize) -> Self {
        assert!(grid_size >= 2, "grid size must be at least 2x2");

        let ball_count = usize::from(scene == DemoScene::FallingBalls) * 3;
        let mut bodies = Vec::with_capacity(3 * grid_size * grid_size - 2 * grid_size + ball_count);
        let mut joints = Vec::with_capacity(4 * grid_size * (grid_size - 1));
        let mut hubs = vec![0usize; grid_size * grid_size];
        let mut node_positions = vec![DVec3::ZERO; grid_size * grid_size];
        let hub_rest_points = hub_rest_points();
        let grid_height = if scene == DemoScene::FallingBalls {
            1.70
        } else {
            3.0
        };

        for row in 0..grid_size {
            for column in 0..grid_size {
                let x = (column as f64 - (grid_size - 1) as f64 * 0.5) * GRID_SPACING;
                let z = if scene == DemoScene::FallingBalls {
                    (row as f64 - (grid_size - 1) as f64 * 0.5) * GRID_SPACING
                } else {
                    -(row as f64) * GRID_SPACING
                };
                let position = DVec3::new(x, grid_height, z);
                let fixed = match scene {
                    DemoScene::JointGrid | DemoScene::CylinderDrape => row == 0,
                    DemoScene::FallingBalls => {
                        (row == 0 || row == grid_size - 1)
                            && (column == 0 || column == grid_size - 1)
                    }
                };
                let body_index = bodies.len();
                let node_index = row * grid_size + column;

                node_positions[node_index] = position;
                hubs[node_index] = body_index;
                bodies.push(AffineBody::new(
                    BodyKind::Hub { fixed },
                    hub_rest_points,
                    position,
                    DQuat::IDENTITY,
                    0.18,
                    fixed,
                ));
            }
        }

        for row in 0..grid_size {
            for column in 0..(grid_size - 1) {
                let start = row * grid_size + column;
                let end = start + 1;
                add_rod(
                    &mut bodies,
                    &mut joints,
                    hubs[start],
                    hubs[end],
                    node_positions[start],
                    node_positions[end],
                );
            }
        }

        for row in 0..(grid_size - 1) {
            for column in 0..grid_size {
                let start = row * grid_size + column;
                let end = start + grid_size;
                add_rod(
                    &mut bodies,
                    &mut joints,
                    hubs[start],
                    hubs[end],
                    node_positions[start],
                    node_positions[end],
                );
            }
        }

        debug_assert_eq!(bodies.len(), 3 * grid_size * grid_size - 2 * grid_size);
        debug_assert_eq!(joints.len(), 4 * grid_size * (grid_size - 1));

        let cylinder = (scene == DemoScene::CylinderDrape).then_some(CylinderCollider {
            origin: DVec3::new(0.0, 1.75, cylinder_origin_z(grid_size)),
            axis: DVec3::X,
            radius: CYLINDER_RADIUS as f64,
            length: cylinder_length(grid_size),
        });
        let mut ball_indices = Vec::new();

        if scene == DemoScene::FallingBalls {
            for position in [
                DVec3::new(-1.10, 2.80, -0.80),
                DVec3::new(0.85, 3.15, -0.25),
                DVec3::new(-0.20, 3.50, 1.05),
            ] {
                ball_indices.push(bodies.len());
                bodies.push(AffineBody::new(
                    BodyKind::Ball,
                    ball_rest_points(),
                    position,
                    DQuat::IDENTITY,
                    1.2,
                    false,
                ));
            }
        }

        debug_validate_direct_solver_topology(&bodies, &joints);
        let solver_scratch = SolverScratch::new(bodies.len(), joints.len());

        Self {
            scene,
            grid_size,
            bodies,
            joints,
            cylinder,
            ball_indices,
            solver_scratch,
        }
    }

    fn step(&mut self, dt: f64) {
        for body in &mut self.bodies {
            body.predict(dt);
        }

        prepare_direct_joint_solver(&self.bodies, &self.joints, &mut self.solver_scratch);

        for _ in 0..COROTATED_ITERATIONS {
            for body in &mut self.bodies {
                body.project_corotated_shape();
            }

            for (residual, joint) in self
                .solver_scratch
                .constraint_residual
                .iter_mut()
                .zip(&self.joints)
            {
                *residual = attachment_position(&self.bodies, joint.a)
                    - attachment_position(&self.bodies, joint.b);
            }

            solve_dual_direct(&self.joints, &mut self.solver_scratch);
            apply_joint_correction(
                &mut self.bodies,
                &self.joints,
                &self.solver_scratch.solution,
                &mut self.solver_scratch.hub_weighted_residual,
            );

            for _ in 0..CONTACT_PASSES {
                match self.scene {
                    DemoScene::JointGrid => {}
                    DemoScene::CylinderDrape => {
                        project_cylinder_contacts(
                            &mut self.bodies,
                            self.cylinder.expect("cylinder scene must have a collider"),
                        );
                    }
                    DemoScene::FallingBalls => {
                        project_ball_contacts(
                            &mut self.bodies,
                            &self.ball_indices,
                            &mut self.solver_scratch.contact_proxy_start,
                            &mut self.solver_scratch.contact_proxy_end,
                        );
                    }
                }
            }
        }

        let velocity_scale = velocity_damping_for_dt(dt) / dt;
        for body in &mut self.bodies {
            body.finish_step(velocity_scale);
        }
    }
}

#[derive(Component)]
struct BodyVisual(usize);

#[derive(Component)]
struct SceneVisual;

#[derive(Component)]
struct MainCamera;

#[derive(Component)]
struct PerformanceOverlay;

#[derive(Component)]
struct FixedHzButton(f64);

#[derive(Component)]
struct GridSizeButton(usize);

#[derive(Resource)]
struct VisualAssets {
    hub_mesh: Handle<Mesh>,
    rod_mesh: Handle<Mesh>,
    ball_mesh: Handle<Mesh>,
    cylinder_mesh: Handle<Mesh>,
    hub_material: Handle<StandardMaterial>,
    rod_material: Handle<StandardMaterial>,
    fixed_material: Handle<StandardMaterial>,
    ball_material: Handle<StandardMaterial>,
    cylinder_material: Handle<StandardMaterial>,
}

fn main() {
    let mut app = App::new();
    app.insert_non_send(PhysxDemo::new())
        .insert_resource(ClearColor(Color::srgb(0.012, 0.018, 0.03)))
        .insert_resource(Time::<Fixed>::from_hz(DEFAULT_FIXED_HZ))
        .insert_resource(ActiveDemo {
            scene: DemoScene::JointGrid,
            backend: SimulationBackend::Project,
            grid_size: DEFAULT_GRID_SIZE,
        })
        .insert_resource(NetSimulation::new(DemoScene::JointGrid))
        .init_resource::<SimulationTiming>()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "M-ABD Physics Comparison".into(),
                resolution: WindowResolution::new(1100, 700),
                present_mode: PresentMode::AutoVsync,
                canvas: Some("#bevy-canvas".into()),
                fit_canvas_to_parent: true,
                prevent_default_event_handling: false,
                ..default()
            }),
            ..default()
        }))
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_systems(Startup, setup_scene)
        .add_systems(FixedUpdate, (step_simulation, step_physx_simulation))
        .add_systems(
            Update,
            (
                switch_demo_scene,
                change_fixed_hz,
                style_fixed_hz_buttons,
                style_grid_size_buttons,
                ApplyDeferred,
                refresh_physx_transforms,
                sync_body_visuals,
                sync_physx_visuals,
                update_performance_overlay,
            )
                .chain(),
        );
    app.run();
}

fn setup_scene(
    mut commands: Commands,
    simulation: Res<NetSimulation>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let backend_hint = if PHYSX_AVAILABLE {
        "[B] Backend"
    } else {
        "PhysX disabled"
    };
    let assets = VisualAssets {
        hub_mesh: meshes.add(Sphere::new(HUB_RADIUS)),
        rod_mesh: meshes.add(Cuboid::new(
            GRID_SPACING as f32 * ROD_LENGTH_FACTOR,
            ROD_THICKNESS,
            ROD_THICKNESS,
        )),
        ball_mesh: meshes.add(Sphere::new(BALL_RADIUS)),
        cylinder_mesh: meshes.add(Cylinder::new(CYLINDER_RADIUS, CYLINDER_LENGTH)),
        hub_material: materials.add(StandardMaterial {
            base_color: Color::srgb(0.56, 0.65, 0.76),
            metallic: 0.55,
            perceptual_roughness: 0.28,
            ..default()
        }),
        rod_material: materials.add(StandardMaterial {
            base_color: Color::srgb(0.38, 0.46, 0.58),
            metallic: 0.65,
            perceptual_roughness: 0.24,
            ..default()
        }),
        fixed_material: materials.add(StandardMaterial {
            base_color: Color::srgb(0.04, 0.55, 0.95),
            metallic: 0.25,
            perceptual_roughness: 0.2,
            ..default()
        }),
        ball_material: materials.add(StandardMaterial {
            base_color: Color::srgb(0.96, 0.29, 0.08),
            metallic: 0.08,
            perceptual_roughness: 0.32,
            ..default()
        }),
        cylinder_material: materials.add(StandardMaterial {
            base_color: Color::srgb(0.56, 0.22, 0.07),
            metallic: 0.04,
            perceptual_roughness: 0.62,
            ..default()
        }),
    };

    spawn_scene_visuals(&mut commands, &simulation, &assets);
    commands.insert_resource(assets);

    commands.spawn((
        PointLight {
            intensity: 1_000_000.0,
            range: 30.0,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_xyz(4.5, 8.0, 6.0),
    ));

    commands.spawn((
        Camera3d::default(),
        camera_transform(simulation.scene, simulation.grid_size),
        MainCamera,
    ));

    commands.spawn((
        Text::new(format!(
            "SCENE 1/{DEMO_COUNT}  Joint grid\nBACKEND  PROJECT\nGRID     {DEFAULT_GRID_SIZE}x{DEFAULT_GRID_SIZE}\nFPS       --\nFRAME     -- ms\nSIM STEP  -- ms\nFIXED    {DEFAULT_FIXED_HZ:>5.0} Hz\n280 bodies | 360 joints\n[1] Grid  [2] Cylinder  [3] Balls  {backend_hint}  [R] Reset"
        )),
        TextFont {
            font_size: FontSize::Px(15.0),
            ..default()
        },
        TextColor(Color::srgb(0.86, 0.91, 1.0)),
        TextShadow::default(),
        Node {
            position_type: PositionType::Absolute,
            top: px(16.0),
            right: px(16.0),
            padding: UiRect::axes(px(12.0), px(9.0)),
            border_radius: BorderRadius::all(px(8.0)),
            ..default()
        },
        BackgroundColor(Color::srgba(0.015, 0.025, 0.045, 0.82)),
        PerformanceOverlay,
    ));

    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                right: px(16.0),
                bottom: px(16.0),
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                column_gap: px(8.0),
                padding: UiRect::axes(px(12.0), px(9.0)),
                border_radius: BorderRadius::all(px(8.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.015, 0.025, 0.045, 0.82)),
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("FIXED HZ"),
                TextFont {
                    font_size: FontSize::Px(14.0),
                    ..default()
                },
                TextColor(Color::srgb(0.70, 0.77, 0.89)),
            ));

            for hz in FIXED_HZ_OPTIONS {
                parent
                    .spawn((
                        Button,
                        FixedHzButton(hz),
                        Node {
                            width: px(50.0),
                            height: px(32.0),
                            justify_content: JustifyContent::Center,
                            align_items: AlignItems::Center,
                            border_radius: BorderRadius::all(px(6.0)),
                            ..default()
                        },
                        BackgroundColor(Color::srgb(0.09, 0.14, 0.22)),
                    ))
                    .with_child((
                        Text::new(format!("{hz:.0}")),
                        TextFont {
                            font_size: FontSize::Px(14.0),
                            ..default()
                        },
                        TextColor(Color::srgb(0.88, 0.92, 1.0)),
                    ));
            }
        });

    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: px(16.0),
                bottom: px(16.0),
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                column_gap: px(8.0),
                padding: UiRect::axes(px(12.0), px(9.0)),
                border_radius: BorderRadius::all(px(8.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.015, 0.025, 0.045, 0.82)),
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("GRID"),
                TextFont {
                    font_size: FontSize::Px(14.0),
                    ..default()
                },
                TextColor(Color::srgb(0.70, 0.77, 0.89)),
            ));

            for grid_size in GRID_SIZE_OPTIONS {
                parent
                    .spawn((
                        Button,
                        GridSizeButton(grid_size),
                        Node {
                            width: px(if grid_size == 100 { 76.0 } else { 66.0 }),
                            height: px(32.0),
                            justify_content: JustifyContent::Center,
                            align_items: AlignItems::Center,
                            border_radius: BorderRadius::all(px(6.0)),
                            ..default()
                        },
                        BackgroundColor(Color::srgb(0.09, 0.14, 0.22)),
                    ))
                    .with_child((
                        Text::new(format!("{grid_size}x{grid_size}")),
                        TextFont {
                            font_size: FontSize::Px(14.0),
                            ..default()
                        },
                        TextColor(Color::srgb(0.88, 0.92, 1.0)),
                    ));
            }
        });
}

fn spawn_scene_visuals(commands: &mut Commands, simulation: &NetSimulation, assets: &VisualAssets) {
    for (index, body) in simulation.bodies.iter().enumerate() {
        let (mesh, material) = match body.kind {
            BodyKind::Hub { fixed: true } => {
                (assets.hub_mesh.clone(), assets.fixed_material.clone())
            }
            BodyKind::Hub { fixed: false } => {
                (assets.hub_mesh.clone(), assets.hub_material.clone())
            }
            BodyKind::Rod => (assets.rod_mesh.clone(), assets.rod_material.clone()),
            BodyKind::Ball => (assets.ball_mesh.clone(), assets.ball_material.clone()),
        };

        commands.spawn((
            Mesh3d(mesh),
            MeshMaterial3d(material),
            body_transform(body),
            BodyVisual(index),
            SceneVisual,
        ));
    }

    if let Some(cylinder) = simulation.cylinder {
        commands.spawn((
            Mesh3d(assets.cylinder_mesh.clone()),
            MeshMaterial3d(assets.cylinder_material.clone()),
            Transform::from_translation(cylinder.origin.as_vec3())
                .with_rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2))
                .with_scale(Vec3::new(
                    1.0,
                    cylinder.length as f32 / CYLINDER_LENGTH,
                    1.0,
                )),
            SceneVisual,
        ));
    }
}

fn switch_demo_scene(
    keys: Res<ButtonInput<KeyCode>>,
    mut commands: Commands,
    mut active: ResMut<ActiveDemo>,
    mut simulation: ResMut<NetSimulation>,
    mut physx: NonSendMut<PhysxDemo>,
    mut timing: ResMut<SimulationTiming>,
    assets: Res<VisualAssets>,
    grid_buttons: Query<(&Interaction, &GridSizeButton), Changed<Interaction>>,
    scene_visuals: Query<Entity, With<SceneVisual>>,
    mut camera: Query<&mut Transform, With<MainCamera>>,
) {
    let selected_grid_size = grid_buttons.iter().find_map(|(interaction, button)| {
        (*interaction == Interaction::Pressed).then_some(button.0)
    });
    let (target_scene, target_backend, target_grid_size) = if keys.just_pressed(KeyCode::Digit1) {
        (DemoScene::JointGrid, active.backend, active.grid_size)
    } else if keys.just_pressed(KeyCode::Digit2) {
        (DemoScene::CylinderDrape, active.backend, active.grid_size)
    } else if keys.just_pressed(KeyCode::Digit3) {
        (DemoScene::FallingBalls, active.backend, active.grid_size)
    } else if keys.just_pressed(KeyCode::KeyB) && PHYSX_AVAILABLE {
        (active.scene, active.backend.toggled(), active.grid_size)
    } else if keys.just_pressed(KeyCode::KeyR) {
        (active.scene, active.backend, active.grid_size)
    } else if let Some(grid_size) = selected_grid_size {
        if grid_size == active.grid_size {
            return;
        }
        (active.scene, active.backend, grid_size)
    } else {
        return;
    };

    for entity in &scene_visuals {
        commands.entity(entity).despawn();
    }

    let next_simulation = NetSimulation::with_grid_size(target_scene, target_grid_size);
    if target_backend == SimulationBackend::Physx {
        physx.reset(target_scene, target_grid_size);
    } else {
        physx.clear();
    }
    spawn_scene_visuals(&mut commands, &next_simulation, &assets);
    *simulation = next_simulation;
    active.scene = target_scene;
    active.backend = target_backend;
    active.grid_size = target_grid_size;
    *timing = SimulationTiming::default();

    if let Ok(mut transform) = camera.single_mut() {
        *transform = camera_transform(target_scene, target_grid_size);
    }
}

fn change_fixed_hz(
    mut fixed_time: ResMut<Time<Fixed>>,
    buttons: Query<(&Interaction, &FixedHzButton), Changed<Interaction>>,
) {
    for (interaction, button) in &buttons {
        if *interaction == Interaction::Pressed {
            fixed_time.set_timestep_hz(button.0);
        }
    }
}

fn style_fixed_hz_buttons(
    fixed_time: Res<Time<Fixed>>,
    mut buttons: Query<(&Interaction, &FixedHzButton, &mut BackgroundColor)>,
) {
    let current_hz = 1.0 / fixed_time.timestep().as_secs_f64();

    for (interaction, button, mut background) in &mut buttons {
        let selected = (button.0 - current_hz).abs() < 0.5;
        *background = match *interaction {
            Interaction::Pressed => Color::srgb(0.16, 0.58, 0.96).into(),
            Interaction::Hovered => Color::srgb(0.15, 0.27, 0.43).into(),
            Interaction::None if selected => Color::srgb(0.04, 0.48, 0.86).into(),
            Interaction::None => Color::srgb(0.09, 0.14, 0.22).into(),
        };
    }
}

fn style_grid_size_buttons(
    active: Res<ActiveDemo>,
    mut buttons: Query<(&Interaction, &GridSizeButton, &mut BackgroundColor)>,
) {
    for (interaction, button, mut background) in &mut buttons {
        let selected = button.0 == active.grid_size;
        *background = match *interaction {
            Interaction::Pressed => Color::srgb(0.16, 0.58, 0.96).into(),
            Interaction::Hovered => Color::srgb(0.15, 0.27, 0.43).into(),
            Interaction::None if selected => Color::srgb(0.04, 0.48, 0.86).into(),
            Interaction::None => Color::srgb(0.09, 0.14, 0.22).into(),
        };
    }
}

fn camera_transform(scene: DemoScene, grid_size: usize) -> Transform {
    let scale = (grid_span(grid_size) / grid_span(DEFAULT_GRID_SIZE)).max(1.0) as f32;

    match scene {
        DemoScene::JointGrid => scaled_camera(
            Vec3::new(7.2, 4.6, 10.5),
            Vec3::new(0.0, 0.7, -2.2),
            Vec3::new(0.0, 0.7, (-grid_span(grid_size) * 0.5 + 0.275) as f32),
            scale,
        ),
        DemoScene::CylinderDrape => scaled_camera(
            Vec3::new(7.2, 5.0, 10.5),
            Vec3::new(0.0, 1.6, -2.3),
            Vec3::new(0.0, 1.6, cylinder_origin_z(grid_size) as f32),
            scale,
        ),
        DemoScene::FallingBalls => scaled_camera(
            Vec3::new(7.4, 5.7, 8.8),
            Vec3::new(0.0, 1.8, 0.0),
            Vec3::new(0.0, 1.8, 0.0),
            scale,
        ),
    }
}

fn scaled_camera(base_position: Vec3, base_target: Vec3, target: Vec3, scale: f32) -> Transform {
    Transform::from_translation(target + (base_position - base_target) * scale)
        .looking_at(target, Vec3::Y)
}

fn step_simulation(
    fixed_time: Res<Time<Fixed>>,
    active: Res<ActiveDemo>,
    mut simulation: ResMut<NetSimulation>,
    mut timing: ResMut<SimulationTiming>,
) {
    if active.backend != SimulationBackend::Project {
        return;
    }

    let start = Instant::now();
    simulation.step(fixed_time.timestep().as_secs_f64());
    timing.latest_step_ms = start.elapsed().as_secs_f64() * 1_000.0;
}

fn step_physx_simulation(
    fixed_time: Res<Time<Fixed>>,
    active: Res<ActiveDemo>,
    mut physx: NonSendMut<PhysxDemo>,
    mut timing: ResMut<SimulationTiming>,
) {
    if active.backend != SimulationBackend::Physx {
        return;
    }

    let start = Instant::now();
    physx.step(fixed_time.timestep().as_secs_f32());
    timing.latest_step_ms = start.elapsed().as_secs_f64() * 1_000.0;
}

fn sync_body_visuals(
    active: Res<ActiveDemo>,
    simulation: Res<NetSimulation>,
    mut visuals: Query<(&BodyVisual, &mut Transform)>,
) {
    if active.backend != SimulationBackend::Project {
        return;
    }

    for (visual, mut transform) in &mut visuals {
        if let Some(body) = simulation.bodies.get(visual.0) {
            *transform = body_transform(body);
        }
    }
}

fn refresh_physx_transforms(active: Res<ActiveDemo>, mut physx: NonSendMut<PhysxDemo>) {
    if active.backend == SimulationBackend::Physx {
        physx.refresh_body_transforms();
    }
}

fn sync_physx_visuals(
    active: Res<ActiveDemo>,
    physx: NonSend<PhysxDemo>,
    mut visuals: Query<(&BodyVisual, &mut Transform)>,
) {
    if active.backend != SimulationBackend::Physx {
        return;
    }

    for (visual, mut transform) in &mut visuals {
        if let Some(body_transform) = physx.body_transforms().get(visual.0) {
            *transform = *body_transform;
        }
    }
}

fn update_performance_overlay(
    diagnostics: Res<DiagnosticsStore>,
    fixed_time: Res<Time<Fixed>>,
    active: Res<ActiveDemo>,
    simulation: Res<NetSimulation>,
    physx: NonSend<PhysxDemo>,
    timing: Res<SimulationTiming>,
    mut overlays: Query<&mut Text, With<PerformanceOverlay>>,
) {
    let fps = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|diagnostic| diagnostic.smoothed())
        .map(|value| format!("{value:>5.1}"))
        .unwrap_or_else(|| "   --".into());
    let frame_time = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FRAME_TIME)
        .and_then(|diagnostic| diagnostic.smoothed())
        .map(|value| format!("{value:>5.2}"))
        .unwrap_or_else(|| "   --".into());
    let fixed_hz = 1.0 / fixed_time.timestep().as_secs_f64();
    let backend = match active.backend {
        SimulationBackend::Project => "PROJECT".to_owned(),
        SimulationBackend::Physx => format!("PHYSX {}", physx.status()),
    };
    let backend_hint = if PHYSX_AVAILABLE {
        "[B] Backend"
    } else {
        "PhysX disabled"
    };

    for mut text in &mut overlays {
        text.0 = format!(
            "SCENE {}/{}  {}\nBACKEND  {backend}\nGRID     {}x{}\nFPS      {fps}\nFRAME    {frame_time} ms\nSIM STEP {:>5.2} ms\nFIXED    {fixed_hz:>5.0} Hz\n{} bodies | {} joints\n[1] Grid  [2] Cylinder  [3] Balls  {backend_hint}  [R] Reset",
            active.scene.number(),
            DEMO_COUNT,
            active.scene.title(),
            active.grid_size,
            active.grid_size,
            timing.latest_step_ms,
            simulation.bodies.len(),
            simulation.joints.len(),
        );
    }
}

fn body_transform(body: &AffineBody) -> Transform {
    let rotation = body.rotation();
    let rotation_f32 = Mat3::from_cols(
        rotation.x_axis.as_vec3(),
        rotation.y_axis.as_vec3(),
        rotation.z_axis.as_vec3(),
    );

    Transform::from_translation(body.centroid().as_vec3())
        .with_rotation(Quat::from_mat3(&rotation_f32).normalize())
}

fn add_rod(
    bodies: &mut Vec<AffineBody>,
    joints: &mut Vec<BallJoint>,
    start_hub: usize,
    end_hub: usize,
    start: DVec3,
    end: DVec3,
) {
    let direction = (end - start).normalize();
    let rotation = DQuat::from_rotation_arc(DVec3::X, direction);
    let rod_index = bodies.len();

    bodies.push(AffineBody::new(
        BodyKind::Rod,
        rod_rest_points(),
        (start + end) * 0.5,
        rotation,
        0.24,
        false,
    ));

    joints.push(BallJoint {
        a: Attachment {
            body: rod_index,
            weights: ROD_START,
        },
        b: Attachment {
            body: start_hub,
            weights: HUB_CENTER,
        },
    });
    joints.push(BallJoint {
        a: Attachment {
            body: rod_index,
            weights: ROD_END,
        },
        b: Attachment {
            body: end_hub,
            weights: HUB_CENTER,
        },
    });
}

fn project_cylinder_contacts(bodies: &mut [AffineBody], cylinder: CylinderCollider) {
    let axis = cylinder.axis.normalize_or_zero();
    if axis.length_squared() <= CONTACT_EPSILON {
        return;
    }

    for body_index in 0..bodies.len() {
        match bodies[body_index].kind {
            BodyKind::Hub { .. } => project_attachment_against_cylinder(
                bodies,
                Attachment {
                    body: body_index,
                    weights: HUB_CENTER,
                },
                HUB_RADIUS as f64,
                cylinder.origin,
                axis,
                cylinder.radius,
            ),
            BodyKind::Rod => {
                let (start, end) = rod_collider_attachments(body_index);
                let start_position = attachment_position(bodies, start);
                let end_position = attachment_position(bodies, end);
                let start_radial = reject_from_axis(start_position - cylinder.origin, axis);
                let direction = end_position - start_position;
                let direction_radial = reject_from_axis(direction, axis);
                let denominator = direction_radial.length_squared();
                let t = if denominator > CONTACT_EPSILON {
                    (-start_radial.dot(direction_radial) / denominator).clamp(0.0, 1.0)
                } else {
                    0.5
                };
                let radial = start_radial + direction_radial * t;
                let contact_distance = cylinder.radius + ROD_THICKNESS as f64 * 0.5;
                let distance_squared = radial.length_squared();
                if distance_squared < contact_distance * contact_distance {
                    project_penetrating_attachment_against_cylinder(
                        bodies,
                        interpolate_attachment(start, end, t),
                        radial,
                        distance_squared,
                        contact_distance,
                        cylinder.origin,
                        axis,
                    );
                }
            }
            BodyKind::Ball => {}
        }
    }
}

fn project_attachment_against_cylinder(
    bodies: &mut [AffineBody],
    attachment: Attachment,
    proxy_radius: f64,
    cylinder_origin: DVec3,
    cylinder_axis: DVec3,
    cylinder_radius: f64,
) {
    let position = attachment_position(bodies, attachment);
    let radial = reject_from_axis(position - cylinder_origin, cylinder_axis);
    project_attachment_against_cylinder_at_radial(
        bodies,
        attachment,
        radial,
        proxy_radius,
        cylinder_origin,
        cylinder_axis,
        cylinder_radius,
    );
}

fn project_attachment_against_cylinder_at_radial(
    bodies: &mut [AffineBody],
    attachment: Attachment,
    radial: DVec3,
    proxy_radius: f64,
    cylinder_origin: DVec3,
    cylinder_axis: DVec3,
    cylinder_radius: f64,
) {
    let contact_distance = cylinder_radius + proxy_radius;
    let distance_squared = radial.length_squared();
    if distance_squared >= contact_distance * contact_distance {
        return;
    }
    project_penetrating_attachment_against_cylinder(
        bodies,
        attachment,
        radial,
        distance_squared,
        contact_distance,
        cylinder_origin,
        cylinder_axis,
    );
}

fn project_penetrating_attachment_against_cylinder(
    bodies: &mut [AffineBody],
    attachment: Attachment,
    radial: DVec3,
    distance_squared: f64,
    contact_distance: f64,
    cylinder_origin: DVec3,
    cylinder_axis: DVec3,
) {
    let distance = distance_squared.sqrt();
    let penetration = contact_distance - distance;
    let normal = if distance_squared > CONTACT_EPSILON {
        radial / distance
    } else {
        let previous = previous_attachment_position(bodies, attachment);
        let previous_radial = reject_from_axis(previous - cylinder_origin, cylinder_axis);
        safe_normal(radial, previous_radial, perpendicular_to(cylinder_axis))
    };
    project_static_attachment(bodies, attachment, normal, penetration);
}

#[cfg(test)]
fn project_cylinder_contacts_reference(bodies: &mut [AffineBody], cylinder: CylinderCollider) {
    let axis = cylinder.axis.normalize_or_zero();
    if axis.length_squared() <= CONTACT_EPSILON {
        return;
    }

    for body_index in 0..bodies.len() {
        let (attachment, proxy_radius) = match bodies[body_index].kind {
            BodyKind::Hub { .. } => (
                Attachment {
                    body: body_index,
                    weights: HUB_CENTER,
                },
                HUB_RADIUS as f64,
            ),
            BodyKind::Rod => {
                let (start, end) = rod_collider_attachments(body_index);
                let start_position = attachment_position(bodies, start);
                let end_position = attachment_position(bodies, end);
                let start_radial = reject_from_axis(start_position - cylinder.origin, axis);
                let direction_radial = reject_from_axis(end_position - start_position, axis);
                let denominator = direction_radial.length_squared();
                let t = if denominator > CONTACT_EPSILON {
                    (-start_radial.dot(direction_radial) / denominator).clamp(0.0, 1.0)
                } else {
                    0.5
                };
                (
                    interpolate_attachment(start, end, t),
                    ROD_THICKNESS as f64 * 0.5,
                )
            }
            BodyKind::Ball => continue,
        };

        let position = attachment_position(bodies, attachment);
        let radial = reject_from_axis(position - cylinder.origin, axis);
        let contact_distance = cylinder.radius + proxy_radius;
        let distance_squared = radial.length_squared();
        if distance_squared >= contact_distance * contact_distance {
            continue;
        }
        let distance = distance_squared.sqrt();
        let previous = previous_attachment_position(bodies, attachment);
        let previous_radial = reject_from_axis(previous - cylinder.origin, axis);
        let normal = safe_normal(radial, previous_radial, perpendicular_to(axis));
        project_static_attachment(bodies, attachment, normal, contact_distance - distance);
    }
}

fn project_ball_contacts(
    bodies: &mut [AffineBody],
    ball_indices: &[usize],
    proxy_start: &mut [DVec3],
    proxy_end: &mut [DVec3],
) {
    for first in 0..ball_indices.len() {
        for second in (first + 1)..ball_indices.len() {
            let a = Attachment {
                body: ball_indices[first],
                weights: HUB_CENTER,
            };
            let b = Attachment {
                body: ball_indices[second],
                weights: HUB_CENTER,
            };
            let fallback = match (first + second) % 3 {
                0 => DVec3::X,
                1 => DVec3::Y,
                _ => DVec3::Z,
            };
            project_attachment_pair(bodies, a, b, BALL_RADIUS as f64 * 2.0, fallback);
        }
    }

    let net_body_count = ball_indices.first().copied().unwrap_or(bodies.len());
    for body_index in 0..net_body_count {
        refresh_contact_proxy(bodies, body_index, proxy_start, proxy_end);
    }

    for &ball_index in ball_indices {
        let ball_center = Attachment {
            body: ball_index,
            weights: HUB_CENTER,
        };
        let mut sphere_position = attachment_position(bodies, ball_center);

        for net_body_index in 0..net_body_count {
            let contact_applied = match bodies[net_body_index].kind {
                BodyKind::Hub { .. } => project_attachment_pair_at_positions(
                    bodies,
                    ball_center,
                    Attachment {
                        body: net_body_index,
                        weights: HUB_CENTER,
                    },
                    sphere_position,
                    proxy_start[net_body_index],
                    BALL_RADIUS as f64 + HUB_RADIUS as f64,
                    DVec3::Y,
                ),
                BodyKind::Rod => {
                    let (start, end) = rod_collider_attachments(net_body_index);
                    let start_position = proxy_start[net_body_index];
                    let end_position = proxy_end[net_body_index];
                    let direction = end_position - start_position;
                    let denominator = direction.length_squared();
                    let t = if denominator > CONTACT_EPSILON {
                        ((sphere_position - start_position).dot(direction) / denominator)
                            .clamp(0.0, 1.0)
                    } else {
                        0.5
                    };
                    let rod_attachment = interpolate_attachment(start, end, t);

                    project_attachment_pair_at_positions(
                        bodies,
                        ball_center,
                        rod_attachment,
                        sphere_position,
                        start_position + direction * t,
                        BALL_RADIUS as f64 + ROD_THICKNESS as f64 * 0.5,
                        DVec3::Y,
                    )
                }
                BodyKind::Ball => false,
            };

            if contact_applied {
                sphere_position = attachment_position(bodies, ball_center);
                refresh_contact_proxy(bodies, net_body_index, proxy_start, proxy_end);
            }
        }
    }
}

fn refresh_contact_proxy(
    bodies: &[AffineBody],
    body_index: usize,
    proxy_start: &mut [DVec3],
    proxy_end: &mut [DVec3],
) {
    match bodies[body_index].kind {
        BodyKind::Hub { .. } => {
            let center = attachment_position(
                bodies,
                Attachment {
                    body: body_index,
                    weights: HUB_CENTER,
                },
            );
            proxy_start[body_index] = center;
            proxy_end[body_index] = center;
        }
        BodyKind::Rod => {
            let (start, end) = rod_collider_attachments(body_index);
            proxy_start[body_index] = attachment_position(bodies, start);
            proxy_end[body_index] = attachment_position(bodies, end);
        }
        BodyKind::Ball => {}
    }
}

#[cfg(test)]
fn project_ball_contacts_uncached(bodies: &mut [AffineBody], ball_indices: &[usize]) {
    for first in 0..ball_indices.len() {
        for second in (first + 1)..ball_indices.len() {
            let a = Attachment {
                body: ball_indices[first],
                weights: HUB_CENTER,
            };
            let b = Attachment {
                body: ball_indices[second],
                weights: HUB_CENTER,
            };
            let fallback = match (first + second) % 3 {
                0 => DVec3::X,
                1 => DVec3::Y,
                _ => DVec3::Z,
            };
            project_attachment_pair(bodies, a, b, BALL_RADIUS as f64 * 2.0, fallback);
        }
    }

    let net_body_count = ball_indices.first().copied().unwrap_or(bodies.len());
    for &ball_index in ball_indices {
        let ball_center = Attachment {
            body: ball_index,
            weights: HUB_CENTER,
        };

        for net_body_index in 0..net_body_count {
            match bodies[net_body_index].kind {
                BodyKind::Hub { .. } => project_attachment_pair(
                    bodies,
                    ball_center,
                    Attachment {
                        body: net_body_index,
                        weights: HUB_CENTER,
                    },
                    BALL_RADIUS as f64 + HUB_RADIUS as f64,
                    DVec3::Y,
                ),
                BodyKind::Rod => {
                    let (start, end) = rod_collider_attachments(net_body_index);
                    let sphere_position = attachment_position(bodies, ball_center);
                    let start_position = attachment_position(bodies, start);
                    let end_position = attachment_position(bodies, end);
                    let direction = end_position - start_position;
                    let denominator = direction.length_squared();
                    let t = if denominator > CONTACT_EPSILON {
                        ((sphere_position - start_position).dot(direction) / denominator)
                            .clamp(0.0, 1.0)
                    } else {
                        0.5
                    };

                    project_attachment_pair(
                        bodies,
                        ball_center,
                        interpolate_attachment(start, end, t),
                        BALL_RADIUS as f64 + ROD_THICKNESS as f64 * 0.5,
                        DVec3::Y,
                    );
                }
                BodyKind::Ball => {}
            }
        }
    }
}

fn project_attachment_pair(
    bodies: &mut [AffineBody],
    a: Attachment,
    b: Attachment,
    minimum_distance: f64,
    fallback: DVec3,
) {
    if a.body == b.body {
        return;
    }

    let a_position = attachment_position(bodies, a);
    let b_position = attachment_position(bodies, b);
    let _ = project_attachment_pair_at_positions(
        bodies,
        a,
        b,
        a_position,
        b_position,
        minimum_distance,
        fallback,
    );
}

fn project_attachment_pair_at_positions(
    bodies: &mut [AffineBody],
    a: Attachment,
    b: Attachment,
    a_position: DVec3,
    b_position: DVec3,
    minimum_distance: f64,
    fallback: DVec3,
) -> bool {
    if a.body == b.body {
        return false;
    }

    let delta = a_position - b_position;
    let distance_squared = delta.length_squared();
    if distance_squared >= minimum_distance * minimum_distance {
        return false;
    }
    let distance = distance_squared.sqrt();
    let penetration = minimum_distance - distance;
    let normal = if distance_squared > CONTACT_EPSILON {
        delta / distance
    } else {
        let previous_delta =
            previous_attachment_position(bodies, a) - previous_attachment_position(bodies, b);
        safe_normal(delta, previous_delta, fallback)
    };
    let a_inverse_weight = attachment_inverse_weight(bodies, a);
    let b_inverse_weight = attachment_inverse_weight(bodies, b);
    let denominator = a_inverse_weight + b_inverse_weight;
    if denominator <= CONTACT_EPSILON {
        return false;
    }

    let multiplier = penetration / denominator;
    apply_attachment_position_delta(bodies, a, normal, multiplier);
    apply_attachment_position_delta(bodies, b, -normal, multiplier);
    true
}

fn project_static_attachment(
    bodies: &mut [AffineBody],
    attachment: Attachment,
    normal: DVec3,
    penetration: f64,
) {
    let inverse_weight = attachment_inverse_weight(bodies, attachment);
    if inverse_weight <= CONTACT_EPSILON {
        return;
    }

    apply_attachment_position_delta(bodies, attachment, normal, penetration / inverse_weight);
}

fn apply_attachment_position_delta(
    bodies: &mut [AffineBody],
    attachment: Attachment,
    direction: DVec3,
    multiplier: f64,
) {
    let body = &mut bodies[attachment.body];
    if body.fixed {
        return;
    }

    let scale = body.inverse_diagonal * multiplier;
    for (position, weight) in body.positions.iter_mut().zip(attachment.weights) {
        *position += direction * (scale * weight);
    }
}

fn rod_collider_attachments(body: usize) -> (Attachment, Attachment) {
    let trim = (1.0 - ROD_LENGTH_FACTOR as f64) * 0.5;
    let start = Attachment {
        body,
        weights: ROD_START,
    };
    let end = Attachment {
        body,
        weights: ROD_END,
    };
    (
        interpolate_attachment(start, end, trim),
        interpolate_attachment(start, end, 1.0 - trim),
    )
}

fn interpolate_attachment(a: Attachment, b: Attachment, t: f64) -> Attachment {
    debug_assert_eq!(a.body, b.body);
    Attachment {
        body: a.body,
        weights: std::array::from_fn(|index| {
            a.weights[index] + (b.weights[index] - a.weights[index]) * t
        }),
    }
}

fn previous_attachment_position(bodies: &[AffineBody], attachment: Attachment) -> DVec3 {
    weighted_point(
        &bodies[attachment.body].previous_positions,
        attachment.weights,
    )
}

fn reject_from_axis(vector: DVec3, axis: DVec3) -> DVec3 {
    vector - axis * vector.dot(axis)
}

fn perpendicular_to(axis: DVec3) -> DVec3 {
    let candidate = if axis.y.abs() < 0.9 {
        DVec3::Y
    } else {
        DVec3::X
    };
    reject_from_axis(candidate, axis).normalize_or_zero()
}

fn safe_normal(primary: DVec3, secondary: DVec3, fallback: DVec3) -> DVec3 {
    if primary.length_squared() > CONTACT_EPSILON {
        primary.normalize()
    } else if secondary.length_squared() > CONTACT_EPSILON {
        secondary.normalize()
    } else {
        fallback.normalize_or_zero()
    }
}

fn prepare_direct_joint_solver(
    bodies: &[AffineBody],
    joints: &[BallJoint],
    scratch: &mut SolverScratch,
) {
    scratch.hub_coupling.fill(0.0);

    for (index, joint) in joints.iter().enumerate() {
        scratch.joint_rod_inverse_weight[index] =
            attachment_inverse_weight(bodies, joint.a).recip();
        scratch.hub_coupling[joint.b.body] = attachment_inverse_weight(bodies, joint.b);
    }
}

#[cfg(debug_assertions)]
fn debug_validate_direct_solver_topology(bodies: &[AffineBody], joints: &[BallJoint]) {
    let mut rod_endpoint_counts = vec![[0_u8; 2]; bodies.len()];

    for joint in joints {
        debug_assert!(
            matches!(bodies[joint.a.body].kind, BodyKind::Rod) && !bodies[joint.a.body].fixed
        );
        debug_assert!(matches!(bodies[joint.b.body].kind, BodyKind::Hub { .. }));
        debug_assert_eq!(joint.b.weights, HUB_CENTER);

        let endpoint = if joint.a.weights == ROD_START {
            0
        } else {
            debug_assert_eq!(joint.a.weights, ROD_END);
            1
        };
        rod_endpoint_counts[joint.a.body][endpoint] += 1;
    }

    debug_assert_eq!(
        ROD_START
            .iter()
            .zip(ROD_END)
            .map(|(start, end)| start * end)
            .sum::<f64>(),
        0.0
    );
    for (index, body) in bodies.iter().enumerate() {
        if matches!(body.kind, BodyKind::Rod) {
            debug_assert_eq!(rod_endpoint_counts[index], [1, 1]);
        }
    }
}

#[cfg(not(debug_assertions))]
fn debug_validate_direct_solver_topology(_bodies: &[AffineBody], _joints: &[BallJoint]) {}

fn solve_dual_direct(joints: &[BallJoint], scratch: &mut SolverScratch) {
    let SolverScratch {
        constraint_residual: residual,
        solution,
        joint_rod_inverse_weight,
        hub_coupling,
        hub_inverse_rod_weight_sum,
        hub_weighted_residual,
        hub_schur_factor,
        ..
    } = scratch;

    hub_inverse_rod_weight_sum.fill(0.0);
    hub_weighted_residual.fill(DVec3::ZERO);

    for (index, joint) in joints.iter().enumerate() {
        let hub = joint.b.body;
        let inverse_rod_weight = joint_rod_inverse_weight[index];
        hub_inverse_rod_weight_sum[hub] += inverse_rod_weight;
        hub_weighted_residual[hub] += residual[index] * inverse_rod_weight;
    }

    for index in 0..hub_schur_factor.len() {
        let coupling = hub_coupling[index];
        hub_schur_factor[index] = coupling / (1.0 + coupling * hub_inverse_rod_weight_sum[index]);
    }

    for (index, joint) in joints.iter().enumerate() {
        let hub = joint.b.body;
        solution[index] = (residual[index] - hub_weighted_residual[hub] * hub_schur_factor[hub])
            * joint_rod_inverse_weight[index];
    }
}

#[cfg(test)]
fn apply_dual_matrix(
    bodies: &[AffineBody],
    joints: &[BallJoint],
    input: &[DVec3],
    output: &mut [DVec3],
    body_forces: &mut [[DVec3; 4]],
) {
    body_forces.fill([DVec3::ZERO; 4]);

    for (joint, multiplier) in joints.iter().zip(input) {
        accumulate_attachment(&mut body_forces[joint.a.body], joint.a.weights, *multiplier);
        accumulate_attachment(
            &mut body_forces[joint.b.body],
            joint.b.weights,
            -*multiplier,
        );
    }

    for (index, force) in body_forces.iter_mut().enumerate() {
        for point_force in force {
            *point_force *= bodies[index].inverse_diagonal;
        }
    }

    for (index, joint) in joints.iter().enumerate() {
        output[index] = weighted_point(&body_forces[joint.a.body], joint.a.weights)
            - weighted_point(&body_forces[joint.b.body], joint.b.weights);
    }
}

fn apply_joint_correction(
    bodies: &mut [AffineBody],
    joints: &[BallJoint],
    multipliers: &[DVec3],
    hub_forces: &mut [DVec3],
) {
    hub_forces.fill(DVec3::ZERO);

    for (joint, multiplier) in joints.iter().zip(multipliers) {
        let rod = &mut bodies[joint.a.body];
        for (position, weight) in rod.positions.iter_mut().zip(joint.a.weights) {
            let force = *multiplier * weight;
            *position -= force * rod.inverse_diagonal;
        }
        hub_forces[joint.b.body] += -*multiplier * HUB_CENTER[0];
    }

    for (body, force) in bodies.iter_mut().zip(hub_forces.iter()) {
        if body.fixed || !matches!(body.kind, BodyKind::Hub { .. }) {
            continue;
        }
        for position in &mut body.positions {
            *position -= *force * body.inverse_diagonal;
        }
    }
}

#[cfg(test)]
fn apply_joint_correction_generic(
    bodies: &mut [AffineBody],
    joints: &[BallJoint],
    multipliers: &[DVec3],
) {
    let mut body_forces = vec![[DVec3::ZERO; 4]; bodies.len()];

    for (joint, multiplier) in joints.iter().zip(multipliers) {
        accumulate_attachment(&mut body_forces[joint.a.body], joint.a.weights, *multiplier);
        accumulate_attachment(
            &mut body_forces[joint.b.body],
            joint.b.weights,
            -*multiplier,
        );
    }

    for (body, forces) in bodies.iter_mut().zip(body_forces.iter()) {
        if body.fixed {
            continue;
        }
        for (position, force) in body.positions.iter_mut().zip(forces.iter()) {
            *position -= *force * body.inverse_diagonal;
        }
    }
}

fn attachment_position(bodies: &[AffineBody], attachment: Attachment) -> DVec3 {
    weighted_point(&bodies[attachment.body].positions, attachment.weights)
}

fn attachment_inverse_weight(bodies: &[AffineBody], attachment: Attachment) -> f64 {
    let body = &bodies[attachment.body];
    body.inverse_diagonal
        * attachment
            .weights
            .iter()
            .map(|weight| weight * weight)
            .sum::<f64>()
}

fn weighted_point(points: &[DVec3; 4], weights: [f64; 4]) -> DVec3 {
    (0..4).fold(DVec3::ZERO, |sum, index| {
        sum + points[index] * weights[index]
    })
}

#[cfg(test)]
fn accumulate_attachment(points: &mut [DVec3; 4], weights: [f64; 4], value: DVec3) {
    for index in 0..4 {
        points[index] += value * weights[index];
    }
}

fn hub_rest_points() -> [DVec3; 4] {
    let radius = HUB_RADIUS as f64;
    let scale = radius / 3.0_f64.sqrt();
    [
        DVec3::new(1.0, 1.0, 1.0) * scale,
        DVec3::new(1.0, -1.0, -1.0) * scale,
        DVec3::new(-1.0, 1.0, -1.0) * scale,
        DVec3::new(-1.0, -1.0, 1.0) * scale,
    ]
}

fn ball_rest_points() -> [DVec3; 4] {
    let radius = BALL_RADIUS as f64;
    let scale = radius / 3.0_f64.sqrt();
    [
        DVec3::new(1.0, 1.0, 1.0) * scale,
        DVec3::new(1.0, -1.0, -1.0) * scale,
        DVec3::new(-1.0, 1.0, -1.0) * scale,
        DVec3::new(-1.0, -1.0, 1.0) * scale,
    ]
}

fn rod_rest_points() -> [DVec3; 4] {
    let half_length = GRID_SPACING * 0.5;
    let radius = ROD_THICKNESS as f64 * 0.5;
    [
        DVec3::new(-half_length, -radius, -radius),
        DVec3::new(-half_length, radius, radius),
        DVec3::new(half_length, -radius, radius),
        DVec3::new(half_length, radius, -radius),
    ]
}

fn centroid(points: &[DVec3; 4]) -> DVec3 {
    (points[0] + points[1] + points[2] + points[3]) * 0.25
}

fn rod_deformation_gradient(points: &[DVec3; 4]) -> DMat3 {
    let inverse_four_half_length = 1.0 / (2.0 * GRID_SPACING);
    let inverse_four_radius = 1.0 / (2.0 * ROD_THICKNESS as f64);

    DMat3::from_cols(
        (-points[0] - points[1] + points[2] + points[3]) * inverse_four_half_length,
        (-points[0] + points[1] - points[2] + points[3]) * inverse_four_radius,
        (-points[0] + points[1] + points[2] - points[3]) * inverse_four_radius,
    )
}

fn closest_rotation(matrix: DMat3) -> DMat3 {
    let mut rotation = matrix;

    for _ in 0..3 {
        if rotation.determinant().abs() <= 1.0e-12 {
            break;
        }
        rotation = (rotation + rotation.inverse().transpose()) * 0.5;
    }

    let x = rotation.x_axis.normalize_or_zero();
    let mut y = (rotation.y_axis - x * x.dot(rotation.y_axis)).normalize_or_zero();
    if x.length_squared() <= 1.0e-12 || y.length_squared() <= 1.0e-12 {
        return DMat3::IDENTITY;
    }

    let mut z = x.cross(y).normalize_or_zero();
    if z.dot(rotation.z_axis) < 0.0 {
        z = -z;
    }
    y = z.cross(x).normalize_or_zero();

    DMat3::from_cols(x, y, z)
}

#[cfg(test)]
mod tests {
    use super::*;

    const STEP_TIME_BATCHES: usize = 7;
    const STEP_TIME_WARMUP_STEPS: usize = 20;
    const STEP_TIME_MEASURED_STEPS: usize = 10;

    #[test]
    #[ignore = "performance check; run `cargo bench-scenes`"]
    fn step_time_scene_1_joint_grid() {
        report_step_time(DemoScene::JointGrid);
    }

    #[test]
    #[ignore = "performance check; run `cargo bench-scenes`"]
    fn step_time_scene_2_cylinder_drape() {
        report_step_time(DemoScene::CylinderDrape);
    }

    #[test]
    #[ignore = "performance check; run `cargo bench-scenes`"]
    fn step_time_scene_3_falling_balls() {
        report_step_time(DemoScene::FallingBalls);
    }

    #[test]
    fn three_polar_iterations_match_five_iteration_reference() {
        fn closest_rotation_with_iterations(matrix: DMat3, iterations: usize) -> DMat3 {
            let mut rotation = matrix;
            for _ in 0..iterations {
                if rotation.determinant().abs() <= 1.0e-12 {
                    break;
                }
                rotation = (rotation + rotation.inverse().transpose()) * 0.5;
            }

            let x = rotation.x_axis.normalize_or_zero();
            let mut y = (rotation.y_axis - x * x.dot(rotation.y_axis)).normalize_or_zero();
            if x.length_squared() <= 1.0e-12 || y.length_squared() <= 1.0e-12 {
                return DMat3::IDENTITY;
            }
            let mut z = x.cross(y).normalize_or_zero();
            if z.dot(rotation.z_axis) < 0.0 {
                z = -z;
            }
            y = z.cross(x).normalize_or_zero();
            DMat3::from_cols(x, y, z)
        }

        for scene in [
            DemoScene::JointGrid,
            DemoScene::CylinderDrape,
            DemoScene::FallingBalls,
        ] {
            let mut simulation = NetSimulation::new(scene);
            let mut maximum_error = 0.0_f64;
            for _ in 0..150 {
                simulation.step(1.0 / DEFAULT_FIXED_HZ);
                for body in &simulation.bodies {
                    if !matches!(body.kind, BodyKind::Rod) {
                        continue;
                    }
                    let gradient = rod_deformation_gradient(&body.positions);
                    let reference = closest_rotation_with_iterations(gradient, 5);
                    maximum_error = maximum_error.max(
                        closest_rotation_with_iterations(gradient, 3)
                            .to_cols_array()
                            .into_iter()
                            .zip(reference.to_cols_array())
                            .map(|(actual, expected)| (actual - expected).abs())
                            .fold(0.0_f64, f64::max),
                    );
                }
            }
            assert!(
                maximum_error < 2.0e-9,
                "{} three-iteration polar error: {maximum_error}",
                scene.title()
            );
        }
    }

    fn report_step_time(scene: DemoScene) {
        let dt = 1.0 / DEFAULT_FIXED_HZ;
        let mut samples = Vec::with_capacity(STEP_TIME_BATCHES);

        for _ in 0..STEP_TIME_BATCHES {
            let mut simulation = NetSimulation::new(scene);
            for _ in 0..STEP_TIME_WARMUP_STEPS {
                simulation.step(dt);
            }

            let start = Instant::now();
            for _ in 0..STEP_TIME_MEASURED_STEPS {
                simulation.step(dt);
            }
            let ms_per_step =
                start.elapsed().as_secs_f64() * 1_000.0 / STEP_TIME_MEASURED_STEPS as f64;

            let _ = std::hint::black_box(&simulation);
            assert!(simulation_is_finite(&simulation));
            samples.push(ms_per_step);
        }

        samples.sort_by(f64::total_cmp);
        let median_ms = samples[samples.len() / 2];
        let mean_ms = samples.iter().sum::<f64>() / samples.len() as f64;

        println!(
            "STEP_TIME scene={} name=\"{}\" median_ms={median_ms:.4} mean_ms={mean_ms:.4} min_ms={:.4} max_ms={:.4} grid={} batches={} measured_steps={} warmup_steps={} dt={dt:.6}",
            scene.number(),
            scene.title(),
            samples[0],
            samples[samples.len() - 1],
            DEFAULT_GRID_SIZE,
            STEP_TIME_BATCHES,
            STEP_TIME_MEASURED_STEPS,
            STEP_TIME_WARMUP_STEPS,
        );
    }

    #[test]
    fn direct_dual_solver_satisfies_the_constraint_matrix() {
        for scene in [
            DemoScene::JointGrid,
            DemoScene::CylinderDrape,
            DemoScene::FallingBalls,
        ] {
            let mut simulation = NetSimulation::new(scene);
            for body in &mut simulation.bodies {
                body.predict(1.0 / DEFAULT_FIXED_HZ);
            }
            prepare_direct_joint_solver(
                &simulation.bodies,
                &simulation.joints,
                &mut simulation.solver_scratch,
            );

            for (index, residual) in simulation
                .solver_scratch
                .constraint_residual
                .iter_mut()
                .enumerate()
            {
                let x = ((index * 17) % 29) as f64 - 14.0;
                let y = ((index * 31) % 37) as f64 - 18.0;
                let z = ((index * 43) % 47) as f64 - 23.0;
                *residual = DVec3::new(x, y, z) * 0.01;
            }
            let expected = simulation.solver_scratch.constraint_residual.clone();

            solve_dual_direct(&simulation.joints, &mut simulation.solver_scratch);

            let mut actual = vec![DVec3::ZERO; simulation.joints.len()];
            let mut body_forces = vec![[DVec3::ZERO; 4]; simulation.bodies.len()];
            apply_dual_matrix(
                &simulation.bodies,
                &simulation.joints,
                &simulation.solver_scratch.solution,
                &mut actual,
                &mut body_forces,
            );

            let maximum_error = actual
                .iter()
                .zip(&expected)
                .map(|(actual, expected)| (*actual - *expected).length())
                .fold(0.0_f64, f64::max);
            let maximum_magnitude = expected
                .iter()
                .map(|value| value.length())
                .fold(0.0_f64, f64::max);
            assert!(
                maximum_error <= 1.0e-12 * (1.0 + maximum_magnitude),
                "{} direct solve residual: {maximum_error}",
                scene.title()
            );
        }
    }

    #[test]
    fn specialized_joint_correction_matches_generic_scatter() {
        for scene in [DemoScene::JointGrid, DemoScene::FallingBalls] {
            let mut simulation = NetSimulation::new(scene);
            for body in &mut simulation.bodies {
                body.predict(1.0 / DEFAULT_FIXED_HZ);
            }

            let multipliers = (0..simulation.joints.len())
                .map(|index| {
                    let x = ((index * 13) % 19) as f64 - 9.0;
                    let y = ((index * 23) % 31) as f64 - 15.0;
                    let z = ((index * 37) % 41) as f64 - 20.0;
                    DVec3::new(x, y, z) * 0.001
                })
                .collect::<Vec<_>>();
            let mut specialized = simulation.bodies.clone();
            let mut generic = simulation.bodies;

            let mut hub_forces = vec![DVec3::ZERO; specialized.len()];
            apply_joint_correction(
                &mut specialized,
                &simulation.joints,
                &multipliers,
                &mut hub_forces,
            );
            apply_joint_correction_generic(&mut generic, &simulation.joints, &multipliers);

            let maximum_error = specialized
                .iter()
                .zip(&generic)
                .flat_map(|(specialized, generic)| {
                    specialized
                        .positions
                        .iter()
                        .zip(&generic.positions)
                        .map(|(specialized, generic)| (*specialized - *generic).length())
                })
                .fold(0.0_f64, f64::max);
            assert!(
                maximum_error < 1.0e-15,
                "{} specialized correction error: {maximum_error}",
                scene.title()
            );
        }
    }

    #[test]
    fn cached_ball_contacts_match_uncached_ordered_projection() {
        let mut simulation = NetSimulation::new(DemoScene::FallingBalls);
        for _ in 0..20 {
            simulation.step(1.0 / DEFAULT_FIXED_HZ);
        }

        let mut cached = simulation.bodies.clone();
        let mut uncached = simulation.bodies;
        let mut proxy_start = vec![DVec3::ZERO; cached.len()];
        let mut proxy_end = vec![DVec3::ZERO; cached.len()];
        project_ball_contacts(
            &mut cached,
            &simulation.ball_indices,
            &mut proxy_start,
            &mut proxy_end,
        );
        project_ball_contacts_uncached(&mut uncached, &simulation.ball_indices);

        let maximum_error = cached
            .iter()
            .zip(&uncached)
            .flat_map(|(cached, uncached)| {
                cached
                    .positions
                    .iter()
                    .zip(&uncached.positions)
                    .map(|(cached, uncached)| (*cached - *uncached).length())
            })
            .fold(0.0_f64, f64::max);
        assert!(
            maximum_error < 1.0e-14,
            "cached contact projection error: {maximum_error}"
        );
    }

    #[test]
    fn specialized_cylinder_contacts_match_reference_projection() {
        let mut simulation = NetSimulation::new(DemoScene::CylinderDrape);
        for _ in 0..20 {
            simulation.step(1.0 / DEFAULT_FIXED_HZ);
        }

        let cylinder = simulation.cylinder.unwrap();
        let mut specialized = simulation.bodies.clone();
        let mut reference = simulation.bodies;
        project_cylinder_contacts(&mut specialized, cylinder);
        project_cylinder_contacts_reference(&mut reference, cylinder);

        let maximum_error = specialized
            .iter()
            .zip(&reference)
            .flat_map(|(specialized, reference)| {
                specialized
                    .positions
                    .iter()
                    .zip(&reference.positions)
                    .map(|(specialized, reference)| (*specialized - *reference).length())
            })
            .fold(0.0_f64, f64::max);
        assert!(
            maximum_error < 1.0e-12,
            "specialized cylinder projection error: {maximum_error}"
        );
    }

    #[test]
    fn center_attached_bodies_remain_unrotated() {
        for scene in [
            DemoScene::JointGrid,
            DemoScene::CylinderDrape,
            DemoScene::FallingBalls,
        ] {
            let mut simulation = NetSimulation::new(scene);
            for _ in 0..150 {
                simulation.step(1.0 / DEFAULT_FIXED_HZ);
            }

            let mut maximum_shape_error = 0.0_f64;
            let mut maximum_velocity_spread = 0.0_f64;
            for body in &simulation.bodies {
                if !matches!(body.kind, BodyKind::Hub { .. } | BodyKind::Ball) {
                    continue;
                }

                let center = body.centroid();
                for index in 0..4 {
                    maximum_shape_error = maximum_shape_error
                        .max((body.positions[index] - center - body.rest_points[index]).length());
                    maximum_velocity_spread = maximum_velocity_spread
                        .max((body.velocities[index] - body.velocities[0]).length());
                }
            }

            assert!(
                maximum_shape_error < 1.0e-10,
                "{} center-attached shape error: {maximum_shape_error}",
                scene.title()
            );
            assert!(
                maximum_velocity_spread < 1.0e-10,
                "{} center-attached velocity spread: {maximum_velocity_spread}",
                scene.title()
            );
        }
    }

    #[test]
    fn rod_deformation_gradient_matches_generic_affine_form() {
        for scene in [
            DemoScene::JointGrid,
            DemoScene::CylinderDrape,
            DemoScene::FallingBalls,
        ] {
            let mut simulation = NetSimulation::new(scene);
            for _ in 0..20 {
                simulation.step(1.0 / DEFAULT_FIXED_HZ);
            }

            let mut maximum_error = 0.0_f64;
            for body in &simulation.bodies {
                if !matches!(body.kind, BodyKind::Rod) {
                    continue;
                }

                let center = body.centroid();
                let covariance = (0..4).fold(DMat3::ZERO, |sum, index| {
                    let position = body.positions[index] - center;
                    let rest = body.rest_points[index];
                    sum + DMat3::from_cols(position * rest.x, position * rest.y, position * rest.z)
                });
                let rest_covariance = body.rest_points.iter().fold(DMat3::ZERO, |sum, rest| {
                    sum + DMat3::from_cols(*rest * rest.x, *rest * rest.y, *rest * rest.z)
                });
                let expected = covariance * rest_covariance.inverse();
                let actual = rod_deformation_gradient(&body.positions);
                maximum_error = maximum_error.max(
                    actual
                        .to_cols_array()
                        .into_iter()
                        .zip(expected.to_cols_array())
                        .map(|(actual, expected)| (actual - expected).abs())
                        .fold(0.0_f64, f64::max),
                );
            }

            assert!(
                maximum_error < 1.0e-11,
                "{} specialized rod gradient error: {maximum_error}",
                scene.title()
            );
        }
    }

    #[test]
    fn builds_the_papers_10_by_10_topology() {
        let simulation = NetSimulation::new(DemoScene::JointGrid);
        let fixed_hubs = simulation.bodies.iter().filter(|body| body.fixed).count();

        assert_eq!(simulation.bodies.len(), 280);
        assert_eq!(simulation.joints.len(), 360);
        assert_eq!(fixed_hubs, 10);
    }

    #[test]
    fn builds_each_supported_grid_topology() {
        for grid_size in GRID_SIZE_OPTIONS {
            let simulation = NetSimulation::with_grid_size(DemoScene::JointGrid, grid_size);
            let expected_bodies = 3 * grid_size * grid_size - 2 * grid_size;
            let expected_joints = 4 * grid_size * (grid_size - 1);

            assert_eq!(simulation.grid_size, grid_size);
            assert_eq!(simulation.bodies.len(), expected_bodies);
            assert_eq!(simulation.joints.len(), expected_joints);
            assert_eq!(
                simulation.bodies.iter().filter(|body| body.fixed).count(),
                grid_size
            );
        }
    }

    #[test]
    fn ball_joints_remain_closed_under_gravity() {
        let mut simulation = NetSimulation::new(DemoScene::JointGrid);

        for _ in 0..30 {
            simulation.step(1.0 / DEFAULT_FIXED_HZ);
        }

        let maximum_gap = simulation
            .joints
            .iter()
            .map(|joint| {
                (attachment_position(&simulation.bodies, joint.a)
                    - attachment_position(&simulation.bodies, joint.b))
                .length()
            })
            .fold(0.0_f64, f64::max);
        let bottom_hub_height = simulation.bodies
            [DEFAULT_GRID_SIZE * (DEFAULT_GRID_SIZE - 1) + DEFAULT_GRID_SIZE / 2]
            .centroid()
            .y;
        let maximum_shape_error = simulation
            .bodies
            .iter()
            .flat_map(|body| {
                let center = body.centroid();
                let rotation = body.rotation();
                (0..4).map(move |index| {
                    (body.positions[index] - center - rotation * body.rest_points[index]).length()
                })
            })
            .fold(0.0_f64, f64::max);

        assert!(maximum_gap.is_finite());
        assert!(maximum_gap < 1.0e-5, "maximum joint gap: {maximum_gap}");
        assert!(bottom_hub_height < 2.5, "net did not fall under gravity");
        assert!(
            maximum_shape_error < 0.05,
            "maximum affine shape error: {maximum_shape_error}"
        );
    }

    #[test]
    fn builds_all_three_demo_scenes() {
        let grid = NetSimulation::new(DemoScene::JointGrid);
        let cylinder = NetSimulation::new(DemoScene::CylinderDrape);
        let balls = NetSimulation::new(DemoScene::FallingBalls);

        assert_eq!(grid.bodies.len(), 280);
        assert_eq!(grid.bodies.iter().filter(|body| body.fixed).count(), 10);
        assert!(grid.cylinder.is_none());
        assert!(grid.ball_indices.is_empty());

        assert_eq!(cylinder.bodies.len(), 280);
        assert_eq!(cylinder.bodies.iter().filter(|body| body.fixed).count(), 10);
        assert!(cylinder.cylinder.is_some());
        assert!(cylinder.ball_indices.is_empty());

        assert_eq!(balls.bodies.len(), 283);
        assert_eq!(balls.bodies.iter().filter(|body| body.fixed).count(), 4);
        assert!(balls.cylinder.is_none());
        assert_eq!(balls.ball_indices.len(), 3);
        assert_eq!(balls.joints.len(), 360);
    }

    #[test]
    fn supported_fixed_rates_update_time_and_preserve_damping() {
        let expected_one_second_damping = VELOCITY_DAMPING_AT_DEFAULT_HZ.powf(DEFAULT_FIXED_HZ);

        for hz in FIXED_HZ_OPTIONS {
            let mut fixed_time = Time::<Fixed>::from_hz(DEFAULT_FIXED_HZ);
            fixed_time.set_timestep_hz(hz);
            let dt = fixed_time.timestep().as_secs_f64();
            let measured_hz = 1.0 / dt;
            let one_second_damping = velocity_damping_for_dt(dt).powf(measured_hz);

            assert!((measured_hz - hz).abs() < 1.0e-4);
            assert!((one_second_damping - expected_one_second_damping).abs() < 1.0e-9);
        }
    }

    #[test]
    fn contact_scenes_remain_finite_at_supported_fixed_rates() {
        for hz in FIXED_HZ_OPTIONS {
            let dt = 1.0 / hz;
            let step_count = (2.0 * hz) as usize;

            for scene in [DemoScene::CylinderDrape, DemoScene::FallingBalls] {
                let mut simulation = NetSimulation::new(scene);
                for _ in 0..step_count {
                    simulation.step(dt);
                }
                assert!(
                    simulation_is_finite(&simulation),
                    "{} became non-finite at {hz} Hz",
                    scene.title()
                );
            }
        }
    }

    #[test]
    fn pair_contact_separates_two_dynamic_attachments() {
        let mut bodies = vec![
            test_body(DVec3::ZERO, false),
            test_body(DVec3::new(0.5, 0.0, 0.0), false),
        ];
        let first = Attachment {
            body: 0,
            weights: HUB_CENTER,
        };
        let second = Attachment {
            body: 1,
            weights: HUB_CENTER,
        };

        project_attachment_pair(&mut bodies, first, second, 1.0, DVec3::X);

        let delta = attachment_position(&bodies, first) - attachment_position(&bodies, second);
        assert!((delta.length() - 1.0).abs() < 1.0e-10);
        assert!((bodies[0].centroid().x + 0.25).abs() < 1.0e-10);
        assert!((bodies[1].centroid().x - 0.75).abs() < 1.0e-10);
    }

    #[test]
    fn static_cylinder_contact_projects_outward() {
        let cylinder = CylinderCollider {
            origin: DVec3::ZERO,
            axis: DVec3::X,
            radius: 0.7,
            length: 5.8,
        };
        let mut bodies = vec![test_body(DVec3::new(0.0, 0.2, 0.0), false)];

        project_cylinder_contacts(&mut bodies, cylinder);

        let radial = reject_from_axis(bodies[0].centroid(), cylinder.axis).length();
        assert!((radial - (cylinder.radius + HUB_RADIUS as f64)).abs() < 1.0e-10);
    }

    #[test]
    fn sphere_capsule_contact_handles_midpoint_and_endcap() {
        for sphere_position in [DVec3::new(0.0, 0.2, 0.0), DVec3::new(0.35, 0.1, 0.0)] {
            let mut rod = AffineBody::new(
                BodyKind::Rod,
                rod_rest_points(),
                DVec3::ZERO,
                DQuat::IDENTITY,
                1.0,
                false,
            );
            rod.inverse_diagonal = 1.0;
            let mut ball = AffineBody::new(
                BodyKind::Ball,
                ball_rest_points(),
                sphere_position,
                DQuat::IDENTITY,
                1.0,
                false,
            );
            ball.inverse_diagonal = 1.0;
            let mut bodies = vec![rod, ball];

            let mut proxy_start = vec![DVec3::ZERO; bodies.len()];
            let mut proxy_end = vec![DVec3::ZERO; bodies.len()];
            project_ball_contacts(&mut bodies, &[1], &mut proxy_start, &mut proxy_end);

            let separation = sphere_rod_separation(&bodies, 1, 0);
            assert!(separation.abs() < 1.0e-10, "separation: {separation}");
        }
    }

    #[test]
    fn coincident_spheres_use_a_finite_fallback_normal() {
        let mut bodies = (0..2)
            .map(|_| {
                let mut body = AffineBody::new(
                    BodyKind::Ball,
                    ball_rest_points(),
                    DVec3::ZERO,
                    DQuat::IDENTITY,
                    1.0,
                    false,
                );
                body.inverse_diagonal = 1.0;
                body
            })
            .collect::<Vec<_>>();

        let mut proxy_start = vec![DVec3::ZERO; bodies.len()];
        let mut proxy_end = vec![DVec3::ZERO; bodies.len()];
        project_ball_contacts(&mut bodies, &[0, 1], &mut proxy_start, &mut proxy_end);

        let distance = (bodies[0].centroid() - bodies[1].centroid()).length();
        assert!(bodies.iter().all(|body| body.centroid().is_finite()));
        assert!((distance - BALL_RADIUS as f64 * 2.0).abs() < 1.0e-10);
    }

    #[test]
    fn cylinder_scene_settles_without_penetration() {
        let mut simulation = NetSimulation::new(DemoScene::CylinderDrape);
        let cylinder = simulation.cylinder.unwrap();

        for _ in 0..120 {
            simulation.step(1.0 / DEFAULT_FIXED_HZ);
        }

        let minimum_separation = simulation
            .bodies
            .iter()
            .enumerate()
            .filter_map(|(index, body)| {
                cylinder_proxy_separation(&simulation.bodies, index, body, cylinder)
            })
            .fold(f64::INFINITY, f64::min);
        let maximum_gap = maximum_joint_gap(&simulation);

        assert!(simulation_is_finite(&simulation));
        assert!(
            minimum_separation >= -1.0e-6,
            "minimum separation: {minimum_separation}"
        );
        assert!(maximum_gap < 5.0e-3, "maximum joint gap: {maximum_gap}");
    }

    #[test]
    fn falling_balls_are_supported_by_the_corner_pinned_net() {
        let mut simulation = NetSimulation::new(DemoScene::FallingBalls);
        let fixed_positions: Vec<[DVec3; 4]> = simulation
            .bodies
            .iter()
            .filter(|body| body.fixed)
            .map(|body| body.positions)
            .collect();

        for _ in 0..150 {
            simulation.step(1.0 / DEFAULT_FIXED_HZ);
        }

        let lowest_net_height = simulation.bodies[..280]
            .iter()
            .map(AffineBody::centroid)
            .map(|position| position.y)
            .fold(f64::INFINITY, f64::min);
        let lowest_ball_height = simulation
            .ball_indices
            .iter()
            .map(|&index| simulation.bodies[index].centroid().y)
            .fold(f64::INFINITY, f64::min);
        let final_fixed_positions: Vec<[DVec3; 4]> = simulation
            .bodies
            .iter()
            .filter(|body| body.fixed)
            .map(|body| body.positions)
            .collect();
        assert!(simulation_is_finite(&simulation));
        assert_eq!(fixed_positions, final_fixed_positions);
        assert!(
            lowest_ball_height > lowest_net_height - BALL_RADIUS as f64,
            "balls passed through the net: ball={lowest_ball_height}, net={lowest_net_height}"
        );
        let minimum_contact_separation = minimum_ball_contact_separation(&simulation);
        assert!(
            minimum_contact_separation >= -2.0e-2,
            "minimum ball contact separation: {minimum_contact_separation}"
        );
        assert!(maximum_joint_gap(&simulation) < 5.0e-3);
    }

    fn test_body(position: DVec3, fixed: bool) -> AffineBody {
        let mut body = AffineBody::new(
            BodyKind::Hub { fixed },
            hub_rest_points(),
            position,
            DQuat::IDENTITY,
            1.0,
            fixed,
        );
        body.inverse_diagonal = if fixed { 0.0 } else { 1.0 };
        body
    }

    fn maximum_joint_gap(simulation: &NetSimulation) -> f64 {
        simulation
            .joints
            .iter()
            .map(|joint| {
                (attachment_position(&simulation.bodies, joint.a)
                    - attachment_position(&simulation.bodies, joint.b))
                .length()
            })
            .fold(0.0, f64::max)
    }

    fn simulation_is_finite(simulation: &NetSimulation) -> bool {
        simulation.bodies.iter().all(|body| {
            body.positions
                .iter()
                .chain(&body.velocities)
                .all(|value| value.is_finite())
        })
    }

    fn cylinder_proxy_separation(
        bodies: &[AffineBody],
        body_index: usize,
        body: &AffineBody,
        cylinder: CylinderCollider,
    ) -> Option<f64> {
        let (attachment, radius) = match body.kind {
            BodyKind::Hub { .. } => (
                Attachment {
                    body: body_index,
                    weights: HUB_CENTER,
                },
                HUB_RADIUS as f64,
            ),
            BodyKind::Rod => {
                let (start, end) = rod_collider_attachments(body_index);
                let start_position = attachment_position(bodies, start);
                let end_position = attachment_position(bodies, end);
                let start_radial =
                    reject_from_axis(start_position - cylinder.origin, cylinder.axis);
                let direction_radial =
                    reject_from_axis(end_position - start_position, cylinder.axis);
                let denominator = direction_radial.length_squared();
                let t = if denominator > CONTACT_EPSILON {
                    (-start_radial.dot(direction_radial) / denominator).clamp(0.0, 1.0)
                } else {
                    0.5
                };
                (
                    interpolate_attachment(start, end, t),
                    ROD_THICKNESS as f64 * 0.5,
                )
            }
            BodyKind::Ball => return None,
        };
        let radial = reject_from_axis(
            attachment_position(bodies, attachment) - cylinder.origin,
            cylinder.axis,
        )
        .length();
        Some(radial - cylinder.radius - radius)
    }

    fn sphere_rod_separation(bodies: &[AffineBody], sphere: usize, rod: usize) -> f64 {
        let sphere_position = bodies[sphere].centroid();
        let (start, end) = rod_collider_attachments(rod);
        let start_position = attachment_position(bodies, start);
        let end_position = attachment_position(bodies, end);
        let direction = end_position - start_position;
        let denominator = direction.length_squared();
        let t = if denominator > CONTACT_EPSILON {
            ((sphere_position - start_position).dot(direction) / denominator).clamp(0.0, 1.0)
        } else {
            0.5
        };
        let closest = attachment_position(bodies, interpolate_attachment(start, end, t));
        (sphere_position - closest).length() - BALL_RADIUS as f64 - ROD_THICKNESS as f64 * 0.5
    }

    fn minimum_ball_contact_separation(simulation: &NetSimulation) -> f64 {
        let mut minimum = f64::INFINITY;

        for first in 0..simulation.ball_indices.len() {
            let first_index = simulation.ball_indices[first];
            for second in (first + 1)..simulation.ball_indices.len() {
                let second_index = simulation.ball_indices[second];
                minimum = minimum.min(
                    (simulation.bodies[first_index].centroid()
                        - simulation.bodies[second_index].centroid())
                    .length()
                        - BALL_RADIUS as f64 * 2.0,
                );
            }

            for net_index in 0..simulation.ball_indices[0] {
                minimum = minimum.min(match simulation.bodies[net_index].kind {
                    BodyKind::Hub { .. } => {
                        (simulation.bodies[first_index].centroid()
                            - simulation.bodies[net_index].centroid())
                        .length()
                            - BALL_RADIUS as f64
                            - HUB_RADIUS as f64
                    }
                    BodyKind::Rod => {
                        sphere_rod_separation(&simulation.bodies, first_index, net_index)
                    }
                    BodyKind::Ball => f64::INFINITY,
                });
            }
        }

        minimum
    }
}

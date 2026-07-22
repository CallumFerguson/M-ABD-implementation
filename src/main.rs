use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::math::{DMat3, DQuat, DVec3};
use bevy::platform::time::Instant;
use bevy::prelude::*;
use bevy::text::FontSize;
use bevy::time::Fixed;
use bevy::window::{PresentMode, WindowResolution};

const GRID_SIZE: usize = 10;
const GRID_SPACING: f64 = 0.55;
const FIXED_HZ: f64 = 30.0;
const FIXED_DT: f64 = 1.0 / FIXED_HZ;
const GRAVITY: DVec3 = DVec3::new(0.0, -9.81, 0.0);
const VELOCITY_DAMPING: f64 = 0.997;
const AFFINE_STIFFNESS: f64 = 12_000.0;
const DUAL_TOLERANCE: f64 = 1.0e-7;
const MAX_PCG_ITERATIONS: usize = 100;
const COROTATED_ITERATIONS: usize = 24;

const HUB_RADIUS: f32 = 0.075;
const ROD_THICKNESS: f32 = 0.055;

const HUB_CENTER: [f64; 4] = [0.25, 0.25, 0.25, 0.25];
const ROD_START: [f64; 4] = [0.5, 0.5, 0.0, 0.0];
const ROD_END: [f64; 4] = [0.0, 0.0, 0.5, 0.5];

#[derive(Clone, Copy)]
enum BodyKind {
    Hub { fixed: bool },
    Rod,
}

struct AffineBody {
    kind: BodyKind,
    fixed: bool,
    rest_points: [DVec3; 4],
    rest_covariance_inverse: DMat3,
    positions: [DVec3; 4],
    previous_positions: [DVec3; 4],
    predicted_positions: [DVec3; 4],
    velocities: [DVec3; 4],
    mass_per_point: f64,
    stiffness: f64,
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
        let rest_covariance = rest_points
            .iter()
            .fold(DMat3::ZERO, |sum, point| sum + outer(*point, *point));

        Self {
            kind,
            fixed,
            rest_points,
            rest_covariance_inverse: rest_covariance.inverse(),
            positions,
            previous_positions: positions,
            predicted_positions: positions,
            velocities: [DVec3::ZERO; 4],
            mass_per_point: mass / 4.0,
            stiffness: AFFINE_STIFFNESS,
            inverse_diagonal: 0.0,
        }
    }

    fn centroid(&self) -> DVec3 {
        centroid(&self.positions)
    }

    fn rotation(&self) -> DMat3 {
        let center = self.centroid();
        let covariance = (0..4).fold(DMat3::ZERO, |sum, index| {
            sum + outer(self.positions[index] - center, self.rest_points[index])
        });
        closest_rotation(covariance * self.rest_covariance_inverse)
    }

    fn predict(&mut self) {
        self.previous_positions = self.positions;

        if self.fixed {
            self.predicted_positions = self.positions;
            self.velocities = [DVec3::ZERO; 4];
            self.inverse_diagonal = 0.0;
            return;
        }

        let inertia = self.mass_per_point / (FIXED_DT * FIXED_DT);
        self.inverse_diagonal = 1.0 / (inertia + self.stiffness);

        for index in 0..4 {
            self.predicted_positions[index] = self.positions[index]
                + self.velocities[index] * FIXED_DT
                + GRAVITY * (FIXED_DT * FIXED_DT);
            self.positions[index] = self.predicted_positions[index];
        }
    }

    fn project_corotated_shape(&mut self) {
        if self.fixed {
            return;
        }

        let inertia = self.mass_per_point / (FIXED_DT * FIXED_DT);
        let center = self.centroid();
        let rotation = self.rotation();

        for index in 0..4 {
            let rigid_target = center + rotation * self.rest_points[index];
            self.positions[index] = (self.predicted_positions[index] * inertia
                + rigid_target * self.stiffness)
                * self.inverse_diagonal;
        }
    }

    fn finish_step(&mut self) {
        if self.fixed {
            self.velocities = [DVec3::ZERO; 4];
            return;
        }

        for index in 0..4 {
            self.velocities[index] = (self.positions[index] - self.previous_positions[index])
                * (VELOCITY_DAMPING / FIXED_DT);
        }
    }
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

#[derive(Resource)]
struct NetSimulation {
    bodies: Vec<AffineBody>,
    joints: Vec<BallJoint>,
}

#[derive(Resource, Default)]
struct SimulationTiming {
    latest_step_ms: f64,
}

impl NetSimulation {
    fn new() -> Self {
        let mut bodies = Vec::with_capacity(3 * GRID_SIZE * GRID_SIZE - 2 * GRID_SIZE);
        let mut joints = Vec::with_capacity(4 * GRID_SIZE * (GRID_SIZE - 1));
        let mut hubs = [[0usize; GRID_SIZE]; GRID_SIZE];
        let mut node_positions = [[DVec3::ZERO; GRID_SIZE]; GRID_SIZE];
        let hub_rest_points = hub_rest_points();

        for row in 0..GRID_SIZE {
            for column in 0..GRID_SIZE {
                let x = (column as f64 - (GRID_SIZE - 1) as f64 * 0.5) * GRID_SPACING;
                let position = DVec3::new(x, 3.0, -(row as f64) * GRID_SPACING);
                let fixed = row == 0;
                let body_index = bodies.len();

                node_positions[row][column] = position;
                hubs[row][column] = body_index;
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

        for row in 0..GRID_SIZE {
            for column in 0..(GRID_SIZE - 1) {
                add_rod(
                    &mut bodies,
                    &mut joints,
                    hubs[row][column],
                    hubs[row][column + 1],
                    node_positions[row][column],
                    node_positions[row][column + 1],
                );
            }
        }

        for row in 0..(GRID_SIZE - 1) {
            for column in 0..GRID_SIZE {
                add_rod(
                    &mut bodies,
                    &mut joints,
                    hubs[row][column],
                    hubs[row + 1][column],
                    node_positions[row][column],
                    node_positions[row + 1][column],
                );
            }
        }

        debug_assert_eq!(bodies.len(), 280);
        debug_assert_eq!(joints.len(), 360);

        Self { bodies, joints }
    }

    fn step(&mut self) {
        for body in &mut self.bodies {
            body.predict();
        }

        for _ in 0..COROTATED_ITERATIONS {
            for body in &mut self.bodies {
                body.project_corotated_shape();
            }

            let constraint_residual: Vec<DVec3> = self
                .joints
                .iter()
                .map(|joint| {
                    attachment_position(&self.bodies, joint.a)
                        - attachment_position(&self.bodies, joint.b)
                })
                .collect();

            let multipliers = solve_dual_pcg(&self.bodies, &self.joints, &constraint_residual);
            apply_joint_correction(&mut self.bodies, &self.joints, &multipliers);
        }

        for body in &mut self.bodies {
            body.finish_step();
        }
    }
}

#[derive(Component)]
struct BodyVisual(usize);

#[derive(Component)]
struct PerformanceOverlay;

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.012, 0.018, 0.03)))
        .insert_resource(Time::<Fixed>::from_hz(FIXED_HZ))
        .insert_resource(NetSimulation::new())
        .init_resource::<SimulationTiming>()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "M-ABD 10x10 Joint Net".into(),
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
        .add_systems(FixedUpdate, step_simulation)
        .add_systems(Update, (sync_body_visuals, update_performance_overlay))
        .run();
}

fn setup_scene(
    mut commands: Commands,
    simulation: Res<NetSimulation>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let hub_mesh = meshes.add(Sphere::new(HUB_RADIUS));
    let rod_mesh = meshes.add(Cuboid::new(
        GRID_SPACING as f32 * 0.78,
        ROD_THICKNESS,
        ROD_THICKNESS,
    ));

    let hub_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.56, 0.65, 0.76),
        metallic: 0.55,
        perceptual_roughness: 0.28,
        ..default()
    });
    let rod_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.38, 0.46, 0.58),
        metallic: 0.65,
        perceptual_roughness: 0.24,
        ..default()
    });
    let fixed_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.04, 0.55, 0.95),
        metallic: 0.25,
        perceptual_roughness: 0.2,
        ..default()
    });

    for (index, body) in simulation.bodies.iter().enumerate() {
        let (mesh, material) = match body.kind {
            BodyKind::Hub { fixed: true } => (hub_mesh.clone(), fixed_material.clone()),
            BodyKind::Hub { fixed: false } => (hub_mesh.clone(), hub_material.clone()),
            BodyKind::Rod => (rod_mesh.clone(), rod_material.clone()),
        };

        commands.spawn((
            Mesh3d(mesh),
            MeshMaterial3d(material),
            body_transform(body),
            BodyVisual(index),
        ));
    }

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
        Transform::from_xyz(7.2, 3.8, 10.5).looking_at(Vec3::new(0.0, 0.6, -1.8), Vec3::Y),
    ));

    commands.spawn((
        Text::new(
            "FPS       --\nFRAME     -- ms\nSIM STEP  -- ms\nSIM       30 Hz\n280 bodies | 360 joints",
        ),
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
}

fn step_simulation(mut simulation: ResMut<NetSimulation>, mut timing: ResMut<SimulationTiming>) {
    let start = Instant::now();
    simulation.step();
    timing.latest_step_ms = start.elapsed().as_secs_f64() * 1_000.0;
}

fn sync_body_visuals(
    simulation: Res<NetSimulation>,
    mut visuals: Query<(&BodyVisual, &mut Transform)>,
) {
    for (visual, mut transform) in &mut visuals {
        *transform = body_transform(&simulation.bodies[visual.0]);
    }
}

fn update_performance_overlay(
    diagnostics: Res<DiagnosticsStore>,
    simulation: Res<NetSimulation>,
    timing: Res<SimulationTiming>,
    mut overlays: Query<&mut Text, With<PerformanceOverlay>>,
) {
    let Some(fps) = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|diagnostic| diagnostic.smoothed())
    else {
        return;
    };
    let Some(frame_time) = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FRAME_TIME)
        .and_then(|diagnostic| diagnostic.smoothed())
    else {
        return;
    };

    for mut text in &mut overlays {
        text.0 = format!(
            "FPS      {fps:>5.1}\nFRAME    {frame_time:>5.2} ms\nSIM STEP {:>5.2} ms\nSIM      {FIXED_HZ:.0} Hz\n{} bodies | {} joints",
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

fn solve_dual_pcg(
    bodies: &[AffineBody],
    joints: &[BallJoint],
    right_hand_side: &[DVec3],
) -> Vec<DVec3> {
    let mut solution = vec![DVec3::ZERO; joints.len()];
    let mut residual = right_hand_side.to_vec();
    let mut preconditioned = vec![DVec3::ZERO; joints.len()];
    let mut direction = vec![DVec3::ZERO; joints.len()];
    let mut matrix_direction = vec![DVec3::ZERO; joints.len()];
    let mut body_forces = vec![[DVec3::ZERO; 4]; bodies.len()];

    for (index, joint) in joints.iter().enumerate() {
        let diagonal =
            attachment_inverse_weight(bodies, joint.a) + attachment_inverse_weight(bodies, joint.b);
        preconditioned[index] = residual[index] / diagonal.max(1.0e-16);
        direction[index] = preconditioned[index];
    }

    let initial_norm = vector_dot(&residual, &residual).sqrt();
    if initial_norm <= 1.0e-12 {
        return solution;
    }

    let mut residual_dot_preconditioned = vector_dot(&residual, &preconditioned);

    for _ in 0..MAX_PCG_ITERATIONS {
        apply_dual_matrix(
            bodies,
            joints,
            &direction,
            &mut matrix_direction,
            &mut body_forces,
        );

        let denominator = vector_dot(&direction, &matrix_direction);
        if denominator.abs() <= 1.0e-20 {
            break;
        }

        let alpha = residual_dot_preconditioned / denominator;
        for index in 0..joints.len() {
            solution[index] += direction[index] * alpha;
            residual[index] -= matrix_direction[index] * alpha;
        }

        if vector_dot(&residual, &residual).sqrt() <= DUAL_TOLERANCE * initial_norm {
            break;
        }

        for (index, joint) in joints.iter().enumerate() {
            let diagonal = attachment_inverse_weight(bodies, joint.a)
                + attachment_inverse_weight(bodies, joint.b);
            preconditioned[index] = residual[index] / diagonal.max(1.0e-16);
        }

        let next_residual_dot_preconditioned = vector_dot(&residual, &preconditioned);
        let beta = next_residual_dot_preconditioned / residual_dot_preconditioned;
        residual_dot_preconditioned = next_residual_dot_preconditioned;

        for index in 0..joints.len() {
            direction[index] = preconditioned[index] + direction[index] * beta;
        }
    }

    solution
}

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

fn apply_joint_correction(bodies: &mut [AffineBody], joints: &[BallJoint], multipliers: &[DVec3]) {
    let mut body_forces = vec![[DVec3::ZERO; 4]; bodies.len()];

    for (joint, multiplier) in joints.iter().zip(multipliers) {
        accumulate_attachment(&mut body_forces[joint.a.body], joint.a.weights, *multiplier);
        accumulate_attachment(
            &mut body_forces[joint.b.body],
            joint.b.weights,
            -*multiplier,
        );
    }

    for (body, forces) in bodies.iter_mut().zip(body_forces) {
        if body.fixed {
            continue;
        }
        for (position, force) in body.positions.iter_mut().zip(forces) {
            *position -= force * body.inverse_diagonal;
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

fn accumulate_attachment(points: &mut [DVec3; 4], weights: [f64; 4], value: DVec3) {
    for index in 0..4 {
        points[index] += value * weights[index];
    }
}

fn vector_dot(a: &[DVec3], b: &[DVec3]) -> f64 {
    a.iter().zip(b).map(|(left, right)| left.dot(*right)).sum()
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

fn outer(a: DVec3, b: DVec3) -> DMat3 {
    DMat3::from_cols(a * b.x, a * b.y, a * b.z)
}

fn closest_rotation(matrix: DMat3) -> DMat3 {
    let mut rotation = matrix;

    for _ in 0..5 {
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

    #[test]
    fn builds_the_papers_10_by_10_topology() {
        let simulation = NetSimulation::new();
        let fixed_hubs = simulation.bodies.iter().filter(|body| body.fixed).count();

        assert_eq!(simulation.bodies.len(), 280);
        assert_eq!(simulation.joints.len(), 360);
        assert_eq!(fixed_hubs, 10);
    }

    #[test]
    fn ball_joints_remain_closed_under_gravity() {
        let mut simulation = NetSimulation::new();

        for _ in 0..30 {
            simulation.step();
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
        let bottom_hub_height = simulation.bodies[GRID_SIZE * (GRID_SIZE - 1) + GRID_SIZE / 2]
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
}

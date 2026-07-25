use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::math::DVec2;
use bevy::math::{DMat3, DQuat, DVec3};
use bevy::platform::time::Instant;
use bevy::prelude::*;
#[cfg(not(target_arch = "wasm32"))]
use bevy::tasks::ComputeTaskPool;
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
const POLAR_NEWTON_ITERATIONS: usize = 2;
const CONTACT_PASSES: usize = 2;
const DEMO_COUNT: usize = 3;
#[cfg(not(target_arch = "wasm32"))]
const PARALLEL_PROJECTION_BODY_THRESHOLD: usize = 1_500;
#[cfg(not(target_arch = "wasm32"))]
const HIGH_TASK_PROJECTION_BODY_THRESHOLD: usize = 4_000;
#[cfg(not(target_arch = "wasm32"))]
const PARALLEL_CYLINDER_BODY_THRESHOLD: usize = 4_000;
#[cfg(not(target_arch = "wasm32"))]
const PARALLEL_BALL_BOUND_BODY_THRESHOLD: usize = 4_000;
#[cfg(not(target_arch = "wasm32"))]
const BALL_BOUND_BODIES_PER_TASK: usize = 1_024;
#[cfg(all(test, not(target_arch = "wasm32")))]
const PARALLEL_JOINT_THRESHOLD: usize = 15_000;
#[cfg(not(target_arch = "wasm32"))]
const PARALLEL_CORRECTION_JOINT_THRESHOLD: usize = 8_000;
const CONTACT_PROXY_CHUNK_SIZE: usize = 32;
const ROD_COLLIDER_TRIM: f64 = (1.0 - ROD_LENGTH_FACTOR as f64) * 0.5;

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
    positions: [DVec3; 4],
    predicted_positions: [DVec3; 4],
    inertia: f64,
    inverse_diagonal: f64,
}

impl AffineBody {
    fn new(
        kind: BodyKind,
        rest_points: [DVec3; 4],
        translation: DVec3,
        rotation: DQuat,
        fixed: bool,
    ) -> Self {
        let positions = rest_points.map(|point| translation + rotation * point);

        Self {
            kind,
            fixed,
            positions,
            predicted_positions: positions,
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

    fn update_time_step_coefficients(&mut self, dt: f64) {
        if self.fixed {
            self.inertia = 0.0;
            self.inverse_diagonal = 0.0;
            return;
        }

        self.inertia = body_mass_per_point(self.kind) / (dt * dt);
        self.inverse_diagonal = 1.0 / (self.inertia + AFFINE_STIFFNESS);
    }

    #[cfg(test)]
    fn predict(&mut self, previous_positions: &mut [DVec3; 4], dt: f64) {
        self.update_time_step_coefficients(dt);
        self.predict_positions(previous_positions, dt, 0.0);
    }

    fn predict_positions(
        &mut self,
        previous_positions: &mut [DVec3; 4],
        dt: f64,
        previous_velocity_scale: f64,
    ) {
        if self.fixed {
            *previous_positions = self.positions;
            self.predicted_positions = self.positions;
            return;
        }

        for index in 0..4 {
            let position = self.positions[index];
            let velocity = (position - previous_positions[index]) * previous_velocity_scale;
            previous_positions[index] = position;
            self.predicted_positions[index] = position + velocity * dt + GRAVITY * (dt * dt);
            self.positions[index] = self.predicted_positions[index];
        }
    }

    #[cfg(test)]
    fn project_corotated_shape(&mut self) {
        self.project_corotated_shape_with_polar_iterations::<POLAR_NEWTON_ITERATIONS>();
    }

    #[cfg(test)]
    fn project_corotated_shape_with_polar_iterations<const POLAR_ITERATIONS: usize>(&mut self) {
        match self.kind {
            BodyKind::Hub { .. } => self.project_hub_shape(),
            BodyKind::Rod => self.project_rod_shape::<POLAR_ITERATIONS, true, true>(),
            BodyKind::Ball => self.project_ball_shape(),
        }
    }

    #[inline]
    fn project_rod_shape<
        const POLAR_ITERATIONS: usize,
        const DIVISION_FREE_FINAL: bool,
        const FUSED_GEOMETRY: bool,
    >(
        &mut self,
    ) {
        let prediction_weight = self.inertia * self.inverse_diagonal;
        let shape_weight = AFFINE_STIFFNESS * self.inverse_diagonal;
        let (center, gradient) = if FUSED_GEOMETRY {
            rod_center_and_deformation_gradient(&self.positions)
        } else {
            (self.centroid(), rod_deformation_gradient(&self.positions))
        };
        let rotation = if DIVISION_FREE_FINAL {
            closest_rotation_with_iterations::<POLAR_ITERATIONS>(gradient)
        } else {
            #[cfg(test)]
            {
                closest_rotation_with_divisive_final::<POLAR_ITERATIONS>(gradient)
            }
            #[cfg(not(test))]
            {
                unreachable!("the divisive final polar round is test-only")
            }
        };
        self.project_rod_shape_from_rotation(center, rotation, prediction_weight, shape_weight);
    }

    #[inline]
    fn project_rod_shape_from_rotation(
        &mut self,
        center: DVec3,
        rotation: DMat3,
        prediction_weight: f64,
        shape_weight: f64,
    ) {
        let half_length = GRID_SPACING * 0.5;
        let radius = ROD_THICKNESS as f64 * 0.5;
        let x = rotation.x_axis * half_length;
        let y = rotation.y_axis * radius;
        let z = rotation.z_axis * radius;
        let rigid_offsets = [-x - y - z, -x + y + z, x - y + z, x + y - z];
        for index in 0..4 {
            let rigid_target = center + rigid_offsets[index];
            self.positions[index] =
                self.predicted_positions[index] * prediction_weight + rigid_target * shape_weight;
        }
    }

    #[inline]
    fn project_hub_shape(&mut self) {
        if self.fixed {
            return;
        }
        self.project_center_attached_shape::<false>();
    }

    #[inline]
    fn project_ball_shape(&mut self) {
        self.project_center_attached_shape::<true>();
    }

    #[inline]
    fn project_center_attached_shape<const BALL: bool>(&mut self) {
        let center = self.centroid();
        let prediction_weight = self.inertia * self.inverse_diagonal;
        let shape_weight = AFFINE_STIFFNESS * self.inverse_diagonal;
        let predicted_center = centroid(&self.predicted_positions);
        let projected_center = predicted_center * prediction_weight + center * shape_weight;
        let rest_points = if BALL {
            ball_rest_points()
        } else {
            hub_rest_points()
        };
        for index in 0..4 {
            self.positions[index] = projected_center + rest_points[index];
        }
    }

    #[cfg(test)]
    fn project_corotated_shape_reference(&mut self) {
        if self.fixed {
            return;
        }

        let center = self.centroid();
        let rotation = match self.kind {
            BodyKind::Rod => closest_rotation(rod_deformation_gradient(&self.positions)),
            BodyKind::Hub { .. } | BodyKind::Ball => DMat3::IDENTITY,
        };
        let rest_points = body_rest_points(self.kind);
        for index in 0..4 {
            let rigid_target = center + rotation * rest_points[index];
            self.positions[index] = (self.predicted_positions[index] * self.inertia
                + rigid_target * AFFINE_STIFFNESS)
                * self.inverse_diagonal;
        }
    }
}

#[inline]
fn project_rod_shape_pair(
    first: &mut AffineBody,
    second: &mut AffineBody,
    endpoints: &mut [DVec3],
) {
    debug_assert_eq!(endpoints.len(), 4);
    let first_prediction_weight = first.inertia * first.inverse_diagonal;
    let first_shape_weight = AFFINE_STIFFNESS * first.inverse_diagonal;
    let second_prediction_weight = second.inertia * second.inverse_diagonal;
    let second_shape_weight = AFFINE_STIFFNESS * second.inverse_diagonal;
    let (first_center, first_gradient) = rod_center_and_deformation_gradient(&first.positions);
    let (second_center, second_gradient) = rod_center_and_deformation_gradient(&second.positions);

    if let Some((first_rotation, second_rotation)) =
        closest_rotation_pair_2(first_gradient, second_gradient)
    {
        first.project_rod_shape_from_rotation(
            first_center,
            first_rotation,
            first_prediction_weight,
            first_shape_weight,
        );
        second.project_rod_shape_from_rotation(
            second_center,
            second_rotation,
            second_prediction_weight,
            second_shape_weight,
        );
    } else {
        first.project_rod_shape::<2, true, true>();
        second.project_rod_shape::<2, true, true>();
    }

    endpoints[..2].copy_from_slice(&rod_joint_endpoints(&first.positions));
    endpoints[2..].copy_from_slice(&rod_joint_endpoints(&second.positions));
}

#[inline]
fn project_rod_shapes<
    const POLAR_ITERATIONS: usize,
    const DIVISION_FREE_FINAL: bool,
    const FUSED_ROD_GEOMETRY: bool,
    const PAIR_ROD_POLAR: bool,
>(
    rods: &mut [AffineBody],
    projected_rod_endpoints: &mut [DVec3],
) {
    debug_assert_eq!(projected_rod_endpoints.len(), rods.len() * 2);
    if PAIR_ROD_POLAR && POLAR_ITERATIONS == 2 && DIVISION_FREE_FINAL && FUSED_ROD_GEOMETRY {
        let paired_rod_count = rods.len() & !1;
        let (paired_rods, tail_rods) = rods.split_at_mut(paired_rod_count);
        let (paired_endpoints, tail_endpoints) =
            projected_rod_endpoints.split_at_mut(paired_rod_count * 2);
        for (rod_pair, endpoint_quad) in paired_rods
            .chunks_exact_mut(2)
            .zip(paired_endpoints.chunks_exact_mut(4))
        {
            let (first, second) = rod_pair.split_at_mut(1);
            project_rod_shape_pair(&mut first[0], &mut second[0], endpoint_quad);
        }
        for (rod, endpoints) in tail_rods.iter_mut().zip(tail_endpoints.chunks_exact_mut(2)) {
            rod.project_rod_shape::<POLAR_ITERATIONS, DIVISION_FREE_FINAL, FUSED_ROD_GEOMETRY>();
            endpoints.copy_from_slice(&rod_joint_endpoints(&rod.positions));
        }
        return;
    }

    for (rod, endpoints) in rods
        .iter_mut()
        .zip(projected_rod_endpoints.chunks_exact_mut(2))
    {
        rod.project_rod_shape::<POLAR_ITERATIONS, DIVISION_FREE_FINAL, FUSED_ROD_GEOMETRY>();
        endpoints.copy_from_slice(&rod_joint_endpoints(&rod.positions));
    }
}

fn project_corotated_shapes<
    const POLAR_ITERATIONS: usize,
    const DIVISION_FREE_FINAL: bool,
    const FUSED_ROD_GEOMETRY: bool,
>(
    bodies: &mut [AffineBody],
    grid_size: usize,
    projected_hub_center: &mut [DVec3],
    projected_rod_endpoints: &mut [DVec3],
) {
    let body_count = bodies.len();
    let hub_count = grid_size * grid_size;
    let rod_count = 2 * grid_size * (grid_size - 1);
    debug_assert!(hub_count + rod_count <= body_count);
    debug_assert_eq!(projected_hub_center.len(), hub_count);
    debug_assert_eq!(projected_rod_endpoints.len(), rod_count * 2);
    let (hubs, remaining) = bodies.split_at_mut(hub_count);
    let (rods, balls) = remaining.split_at_mut(rod_count);

    #[cfg(not(target_arch = "wasm32"))]
    if body_count >= PARALLEL_PROJECTION_BODY_THRESHOLD
        && let Some(task_pool) = ComputeTaskPool::try_get()
    {
        let worker_count = task_pool.thread_num();
        let tasks_per_worker = if body_count < HIGH_TASK_PROJECTION_BODY_THRESHOLD {
            2
        } else {
            4
        };
        let total_tasks = worker_count.saturating_mul(tasks_per_worker);
        let hub_tasks = worker_count.min(hubs.len()).max(1);
        let rod_tasks = total_tasks.saturating_sub(hub_tasks).min(rods.len()).max(1);
        if worker_count > 1 {
            let hub_chunk_size = hubs.len().div_ceil(hub_tasks);
            let rod_chunk_size = rods.len().div_ceil(rod_tasks).next_multiple_of(2);
            task_pool.scope(|scope| {
                for (chunk, endpoint_chunk) in rods
                    .chunks_mut(rod_chunk_size)
                    .zip(projected_rod_endpoints.chunks_mut(rod_chunk_size * 2))
                {
                    scope.spawn(async move {
                        project_rod_shapes::<
                            POLAR_ITERATIONS,
                            DIVISION_FREE_FINAL,
                            FUSED_ROD_GEOMETRY,
                            true,
                        >(chunk, endpoint_chunk);
                    });
                }
                for (chunk, center_chunk) in hubs
                    .chunks_mut(hub_chunk_size)
                    .zip(projected_hub_center.chunks_mut(hub_chunk_size))
                {
                    scope.spawn(async move {
                        for (body, center) in chunk.iter_mut().zip(center_chunk) {
                            body.project_hub_shape();
                            *center = hub_attachment_center(&body.positions);
                        }
                    });
                }
            });
            for ball in balls {
                ball.project_ball_shape();
            }
            return;
        }
    }

    for (hub, center) in hubs.iter_mut().zip(projected_hub_center) {
        hub.project_hub_shape();
        *center = hub_attachment_center(&hub.positions);
    }
    project_rod_shapes::<POLAR_ITERATIONS, DIVISION_FREE_FINAL, FUSED_ROD_GEOMETRY, true>(
        rods,
        projected_rod_endpoints,
    );
    for ball in balls {
        ball.project_ball_shape();
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
    previous_positions: Vec<[DVec3; 4]>,
    joints: Vec<BallJoint>,
    cylinder: Option<CylinderCollider>,
    ball_indices: Vec<usize>,
    solver_scratch: SolverScratch,
    coefficient_dt_bits: u64,
    previous_velocity_scale: f64,
}

struct SolverScratch {
    constraint_residual: Vec<DVec3>,
    #[cfg(test)]
    solution: Vec<DVec3>,
    #[cfg(test)]
    joint_hub: Vec<u32>,
    #[cfg(test)]
    rod_inverse_weight: Vec<f64>,
    #[cfg(test)]
    hub_inverse_rod_weight_sum: Vec<f64>,
    hub_weighted_residual: Vec<DVec3>,
    #[cfg(test)]
    hub_schur_factor: Vec<f64>,
    hub_delta_scale: Vec<f64>,
    contact_chunk_min: Vec<DVec3>,
    contact_chunk_max: Vec<DVec3>,
}

impl SolverScratch {
    fn new(body_count: usize, joints: &[BallJoint], hub_count: usize) -> Self {
        debug_assert_eq!(joints.len() % 2, 0);
        Self {
            constraint_residual: vec![DVec3::ZERO; joints.len()],
            #[cfg(test)]
            solution: vec![DVec3::ZERO; joints.len()],
            #[cfg(test)]
            joint_hub: joints
                .iter()
                .map(|joint| u32::try_from(joint.b.body).expect("hub index must fit in u32"))
                .collect(),
            #[cfg(test)]
            rod_inverse_weight: vec![0.0; joints.len() / 2],
            #[cfg(test)]
            hub_inverse_rod_weight_sum: vec![0.0; hub_count],
            hub_weighted_residual: vec![DVec3::ZERO; hub_count],
            #[cfg(test)]
            hub_schur_factor: vec![0.0; hub_count],
            hub_delta_scale: vec![0.0; hub_count],
            contact_chunk_min: vec![DVec3::ZERO; body_count.div_ceil(CONTACT_PROXY_CHUNK_SIZE)],
            contact_chunk_max: vec![DVec3::ZERO; body_count.div_ceil(CONTACT_PROXY_CHUNK_SIZE)],
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
                    false,
                ));
            }
        }

        debug_validate_direct_solver_topology(&bodies, &joints, grid_size);
        let hub_count = grid_size * grid_size;
        let solver_scratch = SolverScratch::new(bodies.len(), &joints, hub_count);
        let previous_positions = bodies.iter().map(|body| body.positions).collect();

        Self {
            scene,
            grid_size,
            bodies,
            previous_positions,
            joints,
            cylinder,
            ball_indices,
            solver_scratch,
            coefficient_dt_bits: u64::MAX,
            previous_velocity_scale: 0.0,
        }
    }

    fn step(&mut self, dt: f64) {
        self.step_with_polar_iterations::<POLAR_NEWTON_ITERATIONS>(dt);
    }

    #[cfg(test)]
    fn predict_for_test(&mut self, dt: f64) {
        for (body, previous_positions) in self.bodies.iter_mut().zip(&mut self.previous_positions) {
            body.predict(previous_positions, dt);
        }
    }

    fn step_with_polar_iterations<const POLAR_ITERATIONS: usize>(&mut self, dt: f64) {
        self.step_with_joint_projection::<POLAR_ITERATIONS, true, true, true, true>(dt);
    }

    #[cfg(test)]
    fn step_with_dual_projection(&mut self, dt: f64) {
        self.step_with_joint_projection::<POLAR_NEWTON_ITERATIONS, false, true, true, true>(dt);
    }

    #[cfg(test)]
    fn step_with_divisive_final_polar_round(&mut self, dt: f64) {
        self.step_with_joint_projection::<POLAR_NEWTON_ITERATIONS, true, false, true, true>(dt);
    }

    #[cfg(test)]
    fn step_with_legacy_rod_geometry(&mut self, dt: f64) {
        self.step_with_joint_projection::<POLAR_NEWTON_ITERATIONS, true, true, false, true>(dt);
    }

    #[cfg(test)]
    fn step_with_pass_major_cylinder_contacts(&mut self, dt: f64) {
        self.step_with_joint_projection::<POLAR_NEWTON_ITERATIONS, true, true, true, false>(dt);
    }

    fn step_with_joint_projection<
        const POLAR_ITERATIONS: usize,
        const DIRECT: bool,
        const DIVISION_FREE_FINAL: bool,
        const FUSED_ROD_GEOMETRY: bool,
        const BODY_MAJOR_CYLINDER: bool,
    >(
        &mut self,
        dt: f64,
    ) {
        let previous_velocity_scale = self.previous_velocity_scale;
        let dt_bits = dt.to_bits();
        if self.coefficient_dt_bits != dt_bits {
            for body in &mut self.bodies {
                body.update_time_step_coefficients(dt);
            }
            prepare_direct_joint_solver(&self.bodies, &self.joints, &mut self.solver_scratch);
            self.coefficient_dt_bits = dt_bits;
            self.previous_velocity_scale = velocity_damping_for_dt(dt) / dt;
        }

        debug_assert_eq!(self.bodies.len(), self.previous_positions.len());
        for (body, previous_positions) in self.bodies.iter_mut().zip(&mut self.previous_positions) {
            body.predict_positions(previous_positions, dt, previous_velocity_scale);
        }

        for _ in 0..COROTATED_ITERATIONS {
            project_corotated_shapes::<POLAR_ITERATIONS, DIVISION_FREE_FINAL, FUSED_ROD_GEOMETRY>(
                &mut self.bodies,
                self.grid_size,
                &mut self.solver_scratch.hub_weighted_residual,
                &mut self.solver_scratch.constraint_residual,
            );

            if DIRECT {
                project_joint_constraints_direct(
                    &mut self.bodies,
                    &mut self.solver_scratch,
                    self.grid_size * self.grid_size,
                    self.grid_size,
                );
            } else {
                #[cfg(test)]
                {
                    finalize_joint_residuals(
                        &mut self.solver_scratch.constraint_residual,
                        &self.solver_scratch.hub_weighted_residual,
                        &self.solver_scratch.joint_hub,
                    );
                    solve_dual_direct(&mut self.solver_scratch, self.grid_size);
                    apply_joint_correction(
                        &mut self.bodies,
                        &self.joints,
                        &self.solver_scratch.solution,
                        &mut self.solver_scratch.hub_weighted_residual,
                        self.grid_size * self.grid_size,
                        self.grid_size,
                    );
                }
                #[cfg(not(test))]
                unreachable!("the legacy dual projection is test-only");
            }

            match self.scene {
                DemoScene::JointGrid => {}
                DemoScene::CylinderDrape => {
                    let cylinder = self.cylinder.expect("cylinder scene must have a collider");
                    if BODY_MAJOR_CYLINDER {
                        project_cylinder_contact_passes(
                            &mut self.bodies,
                            &self.previous_positions,
                            cylinder,
                        );
                    } else {
                        #[cfg(test)]
                        project_cylinder_contact_passes_pass_major_reference(
                            &mut self.bodies,
                            &self.previous_positions,
                            cylinder,
                        );
                        #[cfg(not(test))]
                        unreachable!("the pass-major cylinder projection is test-only");
                    }
                }
                DemoScene::FallingBalls => {
                    project_ball_contact_passes(
                        &mut self.bodies,
                        &self.previous_positions,
                        &self.ball_indices,
                        &mut self.solver_scratch.contact_chunk_min,
                        &mut self.solver_scratch.contact_chunk_max,
                    );
                }
            }
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

fn simulation_task_pool_options() -> TaskPoolOptions {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut options = TaskPoolOptions::default();
        options.io.max_threads = 1;
        options.async_compute.max_threads = 1;
        options
    }
    #[cfg(target_arch = "wasm32")]
    {
        TaskPoolOptions::default()
    }
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
        .add_plugins(
            DefaultPlugins
                .set(TaskPoolPlugin {
                    task_pool_options: simulation_task_pool_options(),
                })
                .set(WindowPlugin {
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
                }),
        )
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

#[cfg(test)]
fn project_cylinder_contacts(
    bodies: &mut [AffineBody],
    previous_positions: &[[DVec3; 4]],
    cylinder: CylinderCollider,
) {
    debug_assert_eq!(cylinder.axis, DVec3::X);
    debug_assert_eq!(bodies.len(), previous_positions.len());

    for body_index in 0..bodies.len() {
        let _ = project_cylinder_contact_at_index(bodies, previous_positions, body_index, cylinder);
    }
}

#[inline]
fn project_cylinder_contact_at_index(
    bodies: &mut [AffineBody],
    previous_positions: &[[DVec3; 4]],
    body_index: usize,
    cylinder: CylinderCollider,
) -> bool {
    match bodies[body_index].kind {
        BodyKind::Hub { .. } => project_attachment_against_cylinder(
            bodies,
            previous_positions,
            Attachment {
                body: body_index,
                weights: HUB_CENTER,
            },
            HUB_RADIUS as f64,
            cylinder.origin,
            cylinder.radius,
        ),
        BodyKind::Rod => {
            let (start, end) = rod_collider_attachments(body_index);
            let start_position = attachment_position(bodies, start);
            let end_position = attachment_position(bodies, end);
            let start_radial = reject_from_x_axis(start_position - cylinder.origin);
            let direction = end_position - start_position;
            let direction_radial = reject_from_x_axis(direction);
            let contact_distance = cylinder.radius + ROD_THICKNESS as f64 * 0.5;
            let radial_midpoint = start_radial + direction_radial * 0.5;
            let radial_half_extents = direction_radial.abs() * 0.5 + DVec3::splat(contact_distance);
            if radial_midpoint.y.abs() > radial_half_extents.y
                || radial_midpoint.z.abs() > radial_half_extents.z
            {
                return false;
            }
            let denominator = direction_radial.length_squared();
            let t = if denominator > CONTACT_EPSILON {
                (-start_radial.dot(direction_radial) / denominator).clamp(0.0, 1.0)
            } else {
                0.5
            };
            let radial = start_radial + direction_radial * t;
            let distance_squared = radial.length_squared();
            if distance_squared < contact_distance * contact_distance {
                project_penetrating_attachment_against_cylinder(
                    bodies,
                    previous_positions,
                    interpolate_attachment(start, end, t),
                    radial,
                    distance_squared,
                    contact_distance,
                    cylinder.origin,
                )
            } else {
                false
            }
        }
        BodyKind::Ball => false,
    }
}

fn project_cylinder_contacts_body_major(
    bodies: &mut [AffineBody],
    previous_positions: &[[DVec3; 4]],
    cylinder: CylinderCollider,
) {
    debug_assert_eq!(cylinder.axis, DVec3::X);
    debug_assert_eq!(bodies.len(), previous_positions.len());
    debug_assert!(CONTACT_PASSES > 0);

    for body_index in 0..bodies.len() {
        if !project_cylinder_contact_at_index(bodies, previous_positions, body_index, cylinder) {
            continue;
        }
        for _ in 1..CONTACT_PASSES {
            if !project_cylinder_contact_at_index(bodies, previous_positions, body_index, cylinder)
            {
                break;
            }
        }
    }
}

fn project_cylinder_contact_passes(
    bodies: &mut [AffineBody],
    previous_positions: &[[DVec3; 4]],
    cylinder: CylinderCollider,
) {
    debug_assert_eq!(bodies.len(), previous_positions.len());
    #[cfg(not(target_arch = "wasm32"))]
    if bodies.len() >= PARALLEL_CYLINDER_BODY_THRESHOLD
        && let Some(task_pool) = ComputeTaskPool::try_get()
    {
        let task_count = task_pool.thread_num().saturating_mul(2).max(1);
        if task_count > 1 {
            let chunk_size = bodies.len().div_ceil(task_count);
            task_pool.scope(|scope| {
                for (chunk, previous_chunk) in bodies
                    .chunks_mut(chunk_size)
                    .zip(previous_positions.chunks(chunk_size))
                {
                    scope.spawn(async move {
                        project_cylinder_contacts_body_major(chunk, previous_chunk, cylinder);
                    });
                }
            });
            return;
        }
    }

    project_cylinder_contacts_body_major(bodies, previous_positions, cylinder);
}

#[cfg(test)]
fn project_cylinder_contact_passes_pass_major_reference(
    bodies: &mut [AffineBody],
    previous_positions: &[[DVec3; 4]],
    cylinder: CylinderCollider,
) {
    debug_assert_eq!(bodies.len(), previous_positions.len());
    #[cfg(not(target_arch = "wasm32"))]
    if bodies.len() >= PARALLEL_CYLINDER_BODY_THRESHOLD
        && let Some(task_pool) = ComputeTaskPool::try_get()
    {
        let task_count = task_pool.thread_num().saturating_mul(2).max(1);
        if task_count > 1 {
            let chunk_size = bodies.len().div_ceil(task_count);
            task_pool.scope(|scope| {
                for (chunk, previous_chunk) in bodies
                    .chunks_mut(chunk_size)
                    .zip(previous_positions.chunks(chunk_size))
                {
                    scope.spawn(async move {
                        for _ in 0..CONTACT_PASSES {
                            project_cylinder_contacts(chunk, previous_chunk, cylinder);
                        }
                    });
                }
            });
            return;
        }
    }

    for _ in 0..CONTACT_PASSES {
        project_cylinder_contacts(bodies, previous_positions, cylinder);
    }
}

fn project_attachment_against_cylinder(
    bodies: &mut [AffineBody],
    previous_positions: &[[DVec3; 4]],
    attachment: Attachment,
    proxy_radius: f64,
    cylinder_origin: DVec3,
    cylinder_radius: f64,
) -> bool {
    let position = attachment_position(bodies, attachment);
    let radial = reject_from_x_axis(position - cylinder_origin);
    project_attachment_against_cylinder_at_radial(
        bodies,
        previous_positions,
        attachment,
        radial,
        proxy_radius,
        cylinder_origin,
        cylinder_radius,
    )
}

fn project_attachment_against_cylinder_at_radial(
    bodies: &mut [AffineBody],
    previous_positions: &[[DVec3; 4]],
    attachment: Attachment,
    radial: DVec3,
    proxy_radius: f64,
    cylinder_origin: DVec3,
    cylinder_radius: f64,
) -> bool {
    let contact_distance = cylinder_radius + proxy_radius;
    let distance_squared = radial.length_squared();
    if distance_squared >= contact_distance * contact_distance {
        return false;
    }
    project_penetrating_attachment_against_cylinder(
        bodies,
        previous_positions,
        attachment,
        radial,
        distance_squared,
        contact_distance,
        cylinder_origin,
    )
}

fn project_penetrating_attachment_against_cylinder(
    bodies: &mut [AffineBody],
    previous_positions: &[[DVec3; 4]],
    attachment: Attachment,
    radial: DVec3,
    distance_squared: f64,
    contact_distance: f64,
    cylinder_origin: DVec3,
) -> bool {
    let distance = distance_squared.sqrt();
    let penetration = contact_distance - distance;
    let normal = if distance_squared > CONTACT_EPSILON {
        radial / distance
    } else {
        let previous = previous_attachment_position(previous_positions, attachment);
        let previous_radial = reject_from_x_axis(previous - cylinder_origin);
        safe_normal(radial, previous_radial, DVec3::Y)
    };
    project_static_attachment(bodies, attachment, normal, penetration)
}

#[cfg(test)]
fn project_cylinder_contacts_reference(
    bodies: &mut [AffineBody],
    previous_positions: &[[DVec3; 4]],
    cylinder: CylinderCollider,
) {
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
        let previous = previous_attachment_position(previous_positions, attachment);
        let previous_radial = reject_from_axis(previous - cylinder.origin, axis);
        let normal = safe_normal(radial, previous_radial, perpendicular_to(axis));
        project_static_attachment(bodies, attachment, normal, contact_distance - distance);
    }
}

fn project_ball_contact_passes(
    bodies: &mut [AffineBody],
    previous_positions: &[[DVec3; 4]],
    ball_indices: &[usize],
    chunk_min: &mut [DVec3],
    chunk_max: &mut [DVec3],
) {
    for pass in 0..CONTACT_PASSES {
        project_ball_pairs(bodies, previous_positions, ball_indices);
        if pass == 0 {
            rebuild_ball_contact_chunk_bounds(bodies, ball_indices, chunk_min, chunk_max);
        }
        project_balls_against_chunked_net(
            bodies,
            previous_positions,
            ball_indices,
            chunk_min,
            chunk_max,
        );
    }
}

#[cfg(test)]
fn project_ball_contacts(
    bodies: &mut [AffineBody],
    previous_positions: &[[DVec3; 4]],
    ball_indices: &[usize],
    chunk_min: &mut [DVec3],
    chunk_max: &mut [DVec3],
) {
    project_ball_pairs(bodies, previous_positions, ball_indices);
    rebuild_ball_contact_chunk_bounds(bodies, ball_indices, chunk_min, chunk_max);
    project_balls_against_chunked_net(
        bodies,
        previous_positions,
        ball_indices,
        chunk_min,
        chunk_max,
    );
}

fn project_ball_pairs(
    bodies: &mut [AffineBody],
    previous_positions: &[[DVec3; 4]],
    ball_indices: &[usize],
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
            project_attachment_pair(
                bodies,
                previous_positions,
                a,
                b,
                BALL_RADIUS as f64 * 2.0,
                fallback,
            );
        }
    }
}

fn rebuild_ball_contact_chunk_bounds(
    bodies: &[AffineBody],
    ball_indices: &[usize],
    chunk_min: &mut [DVec3],
    chunk_max: &mut [DVec3],
) {
    let net_body_count = ball_indices.first().copied().unwrap_or(bodies.len());
    let chunk_count = net_body_count.div_ceil(CONTACT_PROXY_CHUNK_SIZE);
    debug_assert!(chunk_min.len() >= chunk_count);
    debug_assert!(chunk_max.len() >= chunk_count);
    let net_bodies = &bodies[..net_body_count];
    let chunk_min = &mut chunk_min[..chunk_count];
    let chunk_max = &mut chunk_max[..chunk_count];

    #[cfg(not(target_arch = "wasm32"))]
    if net_body_count >= PARALLEL_BALL_BOUND_BODY_THRESHOLD
        && let Some(task_pool) = ComputeTaskPool::try_get()
    {
        let task_count = task_pool
            .thread_num()
            .min(net_body_count.div_ceil(BALL_BOUND_BODIES_PER_TASK))
            .min(chunk_count)
            .max(1);
        if task_count > 1 {
            let chunks_per_task = chunk_count.div_ceil(task_count);
            let bodies_per_task = chunks_per_task * CONTACT_PROXY_CHUNK_SIZE;
            task_pool.scope(|scope| {
                for ((body_chunk, minimum_chunk), maximum_chunk) in net_bodies
                    .chunks(bodies_per_task)
                    .zip(chunk_min.chunks_mut(chunks_per_task))
                    .zip(chunk_max.chunks_mut(chunks_per_task))
                {
                    scope.spawn(async move {
                        rebuild_ball_contact_chunk_bounds_sequential(
                            body_chunk,
                            minimum_chunk,
                            maximum_chunk,
                        );
                    });
                }
            });
            return;
        }
    }

    rebuild_ball_contact_chunk_bounds_sequential(net_bodies, chunk_min, chunk_max);
}

fn rebuild_ball_contact_chunk_bounds_sequential(
    bodies: &[AffineBody],
    chunk_min: &mut [DVec3],
    chunk_max: &mut [DVec3],
) {
    debug_assert_eq!(
        chunk_min.len(),
        bodies.len().div_ceil(CONTACT_PROXY_CHUNK_SIZE)
    );
    debug_assert_eq!(chunk_max.len(), chunk_min.len());
    for ((body_chunk, minimum_output), maximum_output) in bodies
        .chunks(CONTACT_PROXY_CHUNK_SIZE)
        .zip(chunk_min)
        .zip(chunk_max)
    {
        let mut minimum = DVec3::splat(f64::INFINITY);
        let mut maximum = DVec3::splat(f64::NEG_INFINITY);
        for body in body_chunk {
            let (start, end) = broad_contact_proxy(body);
            minimum = minimum.min(start).min(end);
            maximum = maximum.max(start).max(end);
        }
        *minimum_output = minimum;
        *maximum_output = maximum;
    }
}

#[cfg(test)]
fn rebuild_ball_contact_chunk_bounds_reference(
    bodies: &[AffineBody],
    ball_indices: &[usize],
    chunk_min: &mut [DVec3],
    chunk_max: &mut [DVec3],
) {
    let net_body_count = ball_indices.first().copied().unwrap_or(bodies.len());
    let chunk_count = net_body_count.div_ceil(CONTACT_PROXY_CHUNK_SIZE);
    for chunk_index in 0..chunk_count {
        let first_body = chunk_index * CONTACT_PROXY_CHUNK_SIZE;
        let end_body = (first_body + CONTACT_PROXY_CHUNK_SIZE).min(net_body_count);
        let mut minimum = DVec3::splat(f64::INFINITY);
        let mut maximum = DVec3::splat(f64::NEG_INFINITY);
        for body_index in first_body..end_body {
            let (start, end) = broad_contact_proxy(&bodies[body_index]);
            minimum = minimum.min(start).min(end);
            maximum = maximum.max(start).max(end);
        }
        chunk_min[chunk_index] = minimum;
        chunk_max[chunk_index] = maximum;
    }
}

fn project_balls_against_chunked_net(
    bodies: &mut [AffineBody],
    previous_positions: &[[DVec3; 4]],
    ball_indices: &[usize],
    chunk_min: &mut [DVec3],
    chunk_max: &mut [DVec3],
) {
    let net_body_count = ball_indices.first().copied().unwrap_or(bodies.len());
    let chunk_count = net_body_count.div_ceil(CONTACT_PROXY_CHUNK_SIZE);
    let chunk_margin = BALL_RADIUS as f64 + HUB_RADIUS as f64;
    for &ball_index in ball_indices {
        let ball_center = Attachment {
            body: ball_index,
            weights: HUB_CENTER,
        };
        let mut sphere_position = attachment_position(bodies, ball_center);

        for chunk_index in 0..chunk_count {
            if point_outside_expanded_bounds(
                sphere_position,
                chunk_min[chunk_index],
                chunk_max[chunk_index],
                chunk_margin,
            ) {
                continue;
            }

            let first_body = chunk_index * CONTACT_PROXY_CHUNK_SIZE;
            let end_body = (first_body + CONTACT_PROXY_CHUNK_SIZE).min(net_body_count);
            for net_body_index in first_body..end_body {
                let (start_position, end_position) =
                    detailed_contact_proxy(&bodies[net_body_index]);
                let contact_applied = match bodies[net_body_index].kind {
                    BodyKind::Hub { .. } => project_attachment_pair_at_positions(
                        bodies,
                        previous_positions,
                        ball_center,
                        Attachment {
                            body: net_body_index,
                            weights: HUB_CENTER,
                        },
                        sphere_position,
                        start_position,
                        chunk_margin,
                        DVec3::Y,
                    ),
                    BodyKind::Rod => {
                        let (start, end) = rod_collider_attachments(net_body_index);
                        let direction = end_position - start_position;
                        let minimum_distance = BALL_RADIUS as f64 + ROD_THICKNESS as f64 * 0.5;
                        if point_outside_capsule_bounds(
                            sphere_position,
                            start_position,
                            direction,
                            minimum_distance,
                        ) {
                            continue;
                        }
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
                            previous_positions,
                            ball_center,
                            rod_attachment,
                            sphere_position,
                            start_position + direction * t,
                            minimum_distance,
                            DVec3::Y,
                        )
                    }
                    BodyKind::Ball => false,
                };

                if contact_applied {
                    sphere_position = attachment_position(bodies, ball_center);
                    let (updated_start, updated_end) =
                        detailed_contact_proxy(&bodies[net_body_index]);
                    chunk_min[chunk_index] =
                        chunk_min[chunk_index].min(updated_start).min(updated_end);
                    chunk_max[chunk_index] =
                        chunk_max[chunk_index].max(updated_start).max(updated_end);
                }
            }
        }
    }
}

#[inline]
fn point_outside_expanded_bounds(
    point: DVec3,
    minimum: DVec3,
    maximum: DVec3,
    margin: f64,
) -> bool {
    point.x < minimum.x - margin
        || point.x > maximum.x + margin
        || point.y < minimum.y - margin
        || point.y > maximum.y + margin
        || point.z < minimum.z - margin
        || point.z > maximum.z + margin
}

#[inline]
fn point_outside_capsule_bounds(
    point: DVec3,
    segment_start: DVec3,
    segment_direction: DVec3,
    radius: f64,
) -> bool {
    let midpoint = segment_start + segment_direction * 0.5;
    let half_extents = segment_direction.abs() * 0.5 + DVec3::splat(radius);
    let midpoint_delta = (point - midpoint).abs();
    midpoint_delta.x > half_extents.x
        || midpoint_delta.y > half_extents.y
        || midpoint_delta.z > half_extents.z
}

#[inline]
fn broad_contact_proxy(body: &AffineBody) -> (DVec3, DVec3) {
    match body.kind {
        BodyKind::Hub { .. } => {
            let center = hub_attachment_center(&body.positions);
            (center, center)
        }
        BodyKind::Rod => {
            let [start, end] = rod_joint_endpoints(&body.positions);
            (start, end)
        }
        BodyKind::Ball => unreachable!("balls follow all net bodies"),
    }
}

#[inline]
fn detailed_contact_proxy(body: &AffineBody) -> (DVec3, DVec3) {
    match body.kind {
        BodyKind::Hub { .. } => {
            let center = hub_attachment_center(&body.positions);
            (center, center)
        }
        BodyKind::Rod => {
            let [joint_start, joint_end] = rod_joint_endpoints(&body.positions);
            let trim_offset = (joint_end - joint_start) * ROD_COLLIDER_TRIM;
            (joint_start + trim_offset, joint_end - trim_offset)
        }
        BodyKind::Ball => unreachable!("balls follow all net bodies"),
    }
}

#[cfg(test)]
fn rebuild_ball_contact_detailed_chunk_bounds(
    bodies: &[AffineBody],
    ball_indices: &[usize],
    chunk_min: &mut [DVec3],
    chunk_max: &mut [DVec3],
) {
    let net_body_count = ball_indices.first().copied().unwrap_or(bodies.len());
    let chunk_count = net_body_count.div_ceil(CONTACT_PROXY_CHUNK_SIZE);
    for chunk_index in 0..chunk_count {
        let first_body = chunk_index * CONTACT_PROXY_CHUNK_SIZE;
        let end_body = (first_body + CONTACT_PROXY_CHUNK_SIZE).min(net_body_count);
        let mut minimum = DVec3::splat(f64::INFINITY);
        let mut maximum = DVec3::splat(f64::NEG_INFINITY);
        for body in &bodies[first_body..end_body] {
            let (start, end) = detailed_contact_proxy(body);
            minimum = minimum.min(start).min(end);
            maximum = maximum.max(start).max(end);
        }
        chunk_min[chunk_index] = minimum;
        chunk_max[chunk_index] = maximum;
    }
}

#[cfg(test)]
fn project_ball_contact_passes_detailed_bounds_reference(
    bodies: &mut [AffineBody],
    previous_positions: &[[DVec3; 4]],
    ball_indices: &[usize],
    chunk_min: &mut [DVec3],
    chunk_max: &mut [DVec3],
) {
    for pass in 0..CONTACT_PASSES {
        project_ball_pairs(bodies, previous_positions, ball_indices);
        if pass == 0 {
            rebuild_ball_contact_detailed_chunk_bounds(bodies, ball_indices, chunk_min, chunk_max);
        }
        project_balls_against_chunked_net(
            bodies,
            previous_positions,
            ball_indices,
            chunk_min,
            chunk_max,
        );
    }
}

#[cfg(test)]
fn project_ball_contacts_uncached(
    bodies: &mut [AffineBody],
    previous_positions: &[[DVec3; 4]],
    ball_indices: &[usize],
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
            project_attachment_pair(
                bodies,
                previous_positions,
                a,
                b,
                BALL_RADIUS as f64 * 2.0,
                fallback,
            );
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
                    previous_positions,
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
                        previous_positions,
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
    previous_positions: &[[DVec3; 4]],
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
        previous_positions,
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
    previous_positions: &[[DVec3; 4]],
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
        let previous_delta = previous_attachment_position(previous_positions, a)
            - previous_attachment_position(previous_positions, b);
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
) -> bool {
    let inverse_weight = attachment_inverse_weight(bodies, attachment);
    if inverse_weight <= CONTACT_EPSILON || bodies[attachment.body].fixed {
        return false;
    }

    apply_attachment_position_delta(bodies, attachment, normal, penetration / inverse_weight);
    true
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
    let start = Attachment {
        body,
        weights: ROD_START,
    };
    let end = Attachment {
        body,
        weights: ROD_END,
    };
    (
        interpolate_attachment(start, end, ROD_COLLIDER_TRIM),
        interpolate_attachment(start, end, 1.0 - ROD_COLLIDER_TRIM),
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

fn previous_attachment_position(
    previous_positions: &[[DVec3; 4]],
    attachment: Attachment,
) -> DVec3 {
    weighted_point(&previous_positions[attachment.body], attachment.weights)
}

#[cfg(test)]
fn reject_from_axis(vector: DVec3, axis: DVec3) -> DVec3 {
    vector - axis * vector.dot(axis)
}

#[inline]
fn reject_from_x_axis(vector: DVec3) -> DVec3 {
    DVec3::new(0.0, vector.y, vector.z)
}

#[cfg(test)]
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
    let inverse_rod_weight = joints
        .first()
        .map(|joint| (bodies[joint.a.body].inverse_diagonal * 0.5).recip())
        .unwrap_or(0.0);
    #[cfg(debug_assertions)]
    for joint_pair in joints.chunks_exact(2) {
        let weight = (bodies[joint_pair[0].a.body].inverse_diagonal * 0.5).recip();
        debug_assert_eq!(weight.to_bits(), inverse_rod_weight.to_bits());
    }
    scratch.hub_delta_scale.fill(0.0);

    for joint_pair in joints.chunks_exact(2) {
        scratch.hub_delta_scale[joint_pair[0].b.body] += inverse_rod_weight;
        scratch.hub_delta_scale[joint_pair[1].b.body] += inverse_rod_weight;
    }

    for (index, scale) in scratch.hub_delta_scale.iter_mut().enumerate() {
        let coupling = bodies[index].inverse_diagonal * 0.25;
        let factor = coupling / (1.0 + coupling * *scale);
        *scale = inverse_rod_weight * factor;
    }

    #[cfg(test)]
    prepare_direct_joint_solver_reference(bodies, joints, scratch);
}

#[cfg(test)]
fn prepare_direct_joint_solver_reference(
    bodies: &[AffineBody],
    joints: &[BallJoint],
    scratch: &mut SolverScratch,
) {
    scratch.hub_inverse_rod_weight_sum.fill(0.0);

    for (rod, joint_pair) in joints.chunks_exact(2).enumerate() {
        let inverse_rod_weight = (bodies[joint_pair[0].a.body].inverse_diagonal * 0.5).recip();
        scratch.rod_inverse_weight[rod] = inverse_rod_weight;
        scratch.hub_inverse_rod_weight_sum[joint_pair[0].b.body] += inverse_rod_weight;
        scratch.hub_inverse_rod_weight_sum[joint_pair[1].b.body] += inverse_rod_weight;
    }

    for (index, factor) in scratch.hub_schur_factor.iter_mut().enumerate() {
        let coupling = bodies[index].inverse_diagonal * 0.25;
        *factor = coupling / (1.0 + coupling * scratch.hub_inverse_rod_weight_sum[index]);
    }
}

#[cfg(test)]
fn compute_joint_residuals(
    bodies: &[AffineBody],
    joints: &[BallJoint],
    residuals: &mut [DVec3],
    hub_count: usize,
) {
    let rod_count = joints.len() / 2;
    let rods = &bodies[hub_count..hub_count + rod_count];

    #[cfg(not(target_arch = "wasm32"))]
    if joints.len() >= PARALLEL_JOINT_THRESHOLD
        && let Some(task_pool) = ComputeTaskPool::try_get()
    {
        let task_count = task_pool.thread_num().min(rod_count).max(1);
        if task_count > 1 {
            let rods_per_task = rod_count.div_ceil(task_count);
            task_pool.scope(|scope| {
                for (task_index, residual_chunk) in
                    residuals.chunks_mut(rods_per_task * 2).enumerate()
                {
                    let first_rod = task_index * rods_per_task;
                    let end_rod = (first_rod + rods_per_task).min(rod_count);
                    let rod_chunk = &rods[first_rod..end_rod];
                    let joint_chunk = &joints[first_rod * 2..end_rod * 2];
                    scope.spawn(async move {
                        for ((rod, joint_pair), residual_pair) in rod_chunk
                            .iter()
                            .zip(joint_chunk.chunks_exact(2))
                            .zip(residual_chunk.chunks_exact_mut(2))
                        {
                            let positions = &rod.positions;
                            let start_hub = &bodies[joint_pair[0].b.body].positions;
                            let end_hub = &bodies[joint_pair[1].b.body].positions;
                            let start_hub_center = start_hub[0] * 0.25
                                + start_hub[1] * 0.25
                                + start_hub[2] * 0.25
                                + start_hub[3] * 0.25;
                            let end_hub_center = end_hub[0] * 0.25
                                + end_hub[1] * 0.25
                                + end_hub[2] * 0.25
                                + end_hub[3] * 0.25;
                            residual_pair[0] =
                                positions[0] * 0.5 + positions[1] * 0.5 - start_hub_center;
                            residual_pair[1] =
                                positions[2] * 0.5 + positions[3] * 0.5 - end_hub_center;
                        }
                    });
                }
            });
            return;
        }
    }

    for ((rod, joint_pair), residual_pair) in rods
        .iter()
        .zip(joints.chunks_exact(2))
        .zip(residuals.chunks_exact_mut(2))
    {
        let positions = &rod.positions;
        let start_hub = &bodies[joint_pair[0].b.body].positions;
        let end_hub = &bodies[joint_pair[1].b.body].positions;
        let start_hub_center =
            start_hub[0] * 0.25 + start_hub[1] * 0.25 + start_hub[2] * 0.25 + start_hub[3] * 0.25;
        let end_hub_center =
            end_hub[0] * 0.25 + end_hub[1] * 0.25 + end_hub[2] * 0.25 + end_hub[3] * 0.25;
        residual_pair[0] = positions[0] * 0.5 + positions[1] * 0.5 - start_hub_center;
        residual_pair[1] = positions[2] * 0.5 + positions[3] * 0.5 - end_hub_center;
    }
}

#[cfg(test)]
fn finalize_joint_residuals(
    endpoint_or_residual: &mut [DVec3],
    hub_centers: &[DVec3],
    joint_hub: &[u32],
) {
    for (residual, &hub) in endpoint_or_residual.iter_mut().zip(joint_hub) {
        *residual -= hub_centers[hub as usize];
    }
}

#[cfg(test)]
fn compute_joint_residuals_generic(
    bodies: &[AffineBody],
    joints: &[BallJoint],
    residuals: &mut [DVec3],
) {
    for (residual, joint) in residuals.iter_mut().zip(joints) {
        *residual = attachment_position(bodies, joint.a) - attachment_position(bodies, joint.b);
    }
}

#[cfg(debug_assertions)]
fn debug_validate_direct_solver_topology(
    bodies: &[AffineBody],
    joints: &[BallJoint],
    grid_size: usize,
) {
    let mut rod_endpoint_counts = vec![[0_u8; 2]; bodies.len()];
    let hub_count = bodies
        .iter()
        .take_while(|body| matches!(body.kind, BodyKind::Hub { .. }))
        .count();
    debug_assert_eq!(hub_count, grid_size * grid_size);

    debug_assert_eq!(joints.len() % 2, 0);
    let horizontal_rod_count = grid_size * (grid_size - 1);
    for (rod_offset, pair) in joints.chunks_exact(2).enumerate() {
        let rod = hub_count + rod_offset;
        debug_assert_eq!(pair[0].a.body, rod);
        debug_assert_eq!(pair[0].a.weights, ROD_START);
        debug_assert_eq!(pair[1].a.body, rod);
        debug_assert_eq!(pair[1].a.weights, ROD_END);
        let (start_hub, end_hub) = if rod_offset < horizontal_rod_count {
            let row = rod_offset / (grid_size - 1);
            let column = rod_offset % (grid_size - 1);
            let start = row * grid_size + column;
            (start, start + 1)
        } else {
            let vertical_rod = rod_offset - horizontal_rod_count;
            let start = vertical_rod;
            (start, start + grid_size)
        };
        debug_assert_eq!(pair[0].b.body, start_hub);
        debug_assert_eq!(pair[1].b.body, end_hub);
    }

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
fn debug_validate_direct_solver_topology(
    _bodies: &[AffineBody],
    _joints: &[BallJoint],
    _grid_size: usize,
) {
}

#[cfg(test)]
fn solve_dual_direct(scratch: &mut SolverScratch, grid_size: usize) {
    let SolverScratch {
        constraint_residual: residual,
        solution,
        joint_hub,
        rod_inverse_weight,
        hub_weighted_residual,
        hub_schur_factor,
        ..
    } = scratch;

    debug_assert_eq!(hub_weighted_residual.len(), grid_size * grid_size);
    let horizontal_rod_count = grid_size * (grid_size - 1);
    for row in 0..grid_size {
        for column in 0..grid_size {
            let mut weighted_residual = DVec3::ZERO;
            if column > 0 {
                let rod = row * (grid_size - 1) + column - 1;
                weighted_residual += residual[2 * rod + 1] * rod_inverse_weight[rod];
            }
            if column + 1 < grid_size {
                let rod = row * (grid_size - 1) + column;
                weighted_residual += residual[2 * rod] * rod_inverse_weight[rod];
            }
            if row > 0 {
                let rod = horizontal_rod_count + (row - 1) * grid_size + column;
                weighted_residual += residual[2 * rod + 1] * rod_inverse_weight[rod];
            }
            if row + 1 < grid_size {
                let rod = horizontal_rod_count + row * grid_size + column;
                weighted_residual += residual[2 * rod] * rod_inverse_weight[rod];
            }
            hub_weighted_residual[row * grid_size + column] = weighted_residual;
        }
    }

    for (((solution_pair, residual_pair), hub_pair), &inverse_rod_weight) in solution
        .chunks_exact_mut(2)
        .zip(residual.chunks_exact(2))
        .zip(joint_hub.chunks_exact(2))
        .zip(rod_inverse_weight.iter())
    {
        let start_hub = hub_pair[0] as usize;
        solution_pair[0] = (residual_pair[0]
            - hub_weighted_residual[start_hub] * hub_schur_factor[start_hub])
            * inverse_rod_weight;
        let end_hub = hub_pair[1] as usize;
        solution_pair[1] = (residual_pair[1]
            - hub_weighted_residual[end_hub] * hub_schur_factor[end_hub])
            * inverse_rod_weight;
    }
}

#[inline]
fn gather_joint_residual_row<const HAS_UP: bool, const HAS_DOWN: bool>(
    horizontal_endpoint_or_residual: &mut [DVec3],
    up_endpoint_or_residual: &mut [DVec3],
    down_endpoint_or_residual: &mut [DVec3],
    hub_center_or_delta: &mut [DVec3],
    hub_delta_scale: &[f64],
) {
    let grid_size = hub_center_or_delta.len();
    debug_assert!(grid_size >= 2);
    debug_assert_eq!(horizontal_endpoint_or_residual.len(), 2 * (grid_size - 1));
    debug_assert!(!HAS_UP || up_endpoint_or_residual.len() == 2 * grid_size);
    debug_assert!(!HAS_DOWN || down_endpoint_or_residual.len() == 2 * grid_size);
    debug_assert_eq!(hub_delta_scale.len(), grid_size);

    let mut horizontal_pairs = horizontal_endpoint_or_residual.chunks_exact_mut(2);
    let mut up_pairs = up_endpoint_or_residual.chunks_exact_mut(2);
    let mut down_pairs = down_endpoint_or_residual.chunks_exact_mut(2);

    let first_center = hub_center_or_delta[0];
    let first_horizontal_pair = horizontal_pairs
        .next()
        .expect("a joint row must contain a horizontal rod");
    let right_residual = first_horizontal_pair[0] - first_center;
    first_horizontal_pair[0] = right_residual;
    let mut current_center = hub_center_or_delta[1];
    let mut carried_left_residual = first_horizontal_pair[1] - current_center;
    first_horizontal_pair[1] = carried_left_residual;

    let mut residual_sum = DVec3::ZERO;
    residual_sum += right_residual;
    if HAS_UP {
        let up_pair = up_pairs.next().expect("an upper rod must exist per hub");
        let up_residual = up_pair[1] - first_center;
        up_pair[1] = up_residual;
        residual_sum += up_residual;
    }
    if HAS_DOWN {
        let down_pair = down_pairs.next().expect("a lower rod must exist per hub");
        let down_residual = down_pair[0] - first_center;
        down_pair[0] = down_residual;
        residual_sum += down_residual;
    }
    hub_center_or_delta[0] = residual_sum * hub_delta_scale[0];

    for column in 1..grid_size - 1 {
        let horizontal_pair = horizontal_pairs
            .next()
            .expect("a horizontal rod must exist between adjacent hubs");
        let right_residual = horizontal_pair[0] - current_center;
        horizontal_pair[0] = right_residual;
        let next_center = hub_center_or_delta[column + 1];
        let next_left_residual = horizontal_pair[1] - next_center;
        horizontal_pair[1] = next_left_residual;

        let mut residual_sum = DVec3::ZERO;
        residual_sum += carried_left_residual;
        residual_sum += right_residual;
        if HAS_UP {
            let up_pair = up_pairs.next().expect("an upper rod must exist per hub");
            let up_residual = up_pair[1] - current_center;
            up_pair[1] = up_residual;
            residual_sum += up_residual;
        }
        if HAS_DOWN {
            let down_pair = down_pairs.next().expect("a lower rod must exist per hub");
            let down_residual = down_pair[0] - current_center;
            down_pair[0] = down_residual;
            residual_sum += down_residual;
        }
        hub_center_or_delta[column] = residual_sum * hub_delta_scale[column];
        current_center = next_center;
        carried_left_residual = next_left_residual;
    }

    let last_column = grid_size - 1;
    let mut residual_sum = DVec3::ZERO;
    residual_sum += carried_left_residual;
    if HAS_UP {
        let up_pair = up_pairs.next().expect("an upper rod must exist per hub");
        let up_residual = up_pair[1] - current_center;
        up_pair[1] = up_residual;
        residual_sum += up_residual;
    }
    if HAS_DOWN {
        let down_pair = down_pairs.next().expect("a lower rod must exist per hub");
        let down_residual = down_pair[0] - current_center;
        down_pair[0] = down_residual;
        residual_sum += down_residual;
    }
    hub_center_or_delta[last_column] = residual_sum * hub_delta_scale[last_column];

    debug_assert!(horizontal_pairs.next().is_none());
    debug_assert!(!HAS_UP || up_pairs.next().is_none());
    debug_assert!(!HAS_DOWN || down_pairs.next().is_none());
}

fn gather_joint_residuals_row_streaming(
    endpoint_or_residual: &mut [DVec3],
    hub_center_or_delta: &mut [DVec3],
    hub_delta_scale: &[f64],
    grid_size: usize,
) {
    debug_assert!(grid_size >= 2);
    debug_assert_eq!(hub_center_or_delta.len(), grid_size * grid_size);
    debug_assert_eq!(hub_delta_scale.len(), grid_size * grid_size);
    debug_assert_eq!(endpoint_or_residual.len(), 4 * grid_size * (grid_size - 1));

    let horizontal_row_len = 2 * (grid_size - 1);
    let vertical_row_len = 2 * grid_size;
    let horizontal_endpoint_count = horizontal_row_len * grid_size;
    let (horizontal, vertical) = endpoint_or_residual.split_at_mut(horizontal_endpoint_count);

    gather_joint_residual_row::<false, true>(
        &mut horizontal[..horizontal_row_len],
        &mut [],
        &mut vertical[..vertical_row_len],
        &mut hub_center_or_delta[..grid_size],
        &hub_delta_scale[..grid_size],
    );

    for row in 1..grid_size - 1 {
        let horizontal_start = row * horizontal_row_len;
        let hub_start = row * grid_size;
        let (up_and_before, down_and_after) = vertical.split_at_mut(row * vertical_row_len);
        let up = &mut up_and_before[(row - 1) * vertical_row_len..];
        let down = &mut down_and_after[..vertical_row_len];
        gather_joint_residual_row::<true, true>(
            &mut horizontal[horizontal_start..horizontal_start + horizontal_row_len],
            up,
            down,
            &mut hub_center_or_delta[hub_start..hub_start + grid_size],
            &hub_delta_scale[hub_start..hub_start + grid_size],
        );
    }

    let last_row = grid_size - 1;
    let horizontal_start = last_row * horizontal_row_len;
    let hub_start = last_row * grid_size;
    let up_start = (last_row - 1) * vertical_row_len;
    gather_joint_residual_row::<true, false>(
        &mut horizontal[horizontal_start..horizontal_start + horizontal_row_len],
        &mut vertical[up_start..up_start + vertical_row_len],
        &mut [],
        &mut hub_center_or_delta[hub_start..hub_start + grid_size],
        &hub_delta_scale[hub_start..hub_start + grid_size],
    );
}

#[cfg(test)]
fn gather_joint_residuals_reference(
    endpoint_or_residual: &mut [DVec3],
    hub_center_or_delta: &mut [DVec3],
    hub_delta_scale: &[f64],
    grid_size: usize,
) {
    let horizontal_rod_count = grid_size * (grid_size - 1);
    for row in 0..grid_size {
        for column in 0..grid_size {
            let hub_index = row * grid_size + column;
            let center = hub_center_or_delta[hub_index];
            let mut residual_sum = DVec3::ZERO;
            if column > 0 {
                let rod = row * (grid_size - 1) + column - 1;
                let endpoint = 2 * rod + 1;
                let residual = endpoint_or_residual[endpoint] - center;
                endpoint_or_residual[endpoint] = residual;
                residual_sum += residual;
            }
            if column + 1 < grid_size {
                let rod = row * (grid_size - 1) + column;
                let endpoint = 2 * rod;
                let residual = endpoint_or_residual[endpoint] - center;
                endpoint_or_residual[endpoint] = residual;
                residual_sum += residual;
            }
            if row > 0 {
                let rod = horizontal_rod_count + (row - 1) * grid_size + column;
                let endpoint = 2 * rod + 1;
                let residual = endpoint_or_residual[endpoint] - center;
                endpoint_or_residual[endpoint] = residual;
                residual_sum += residual;
            }
            if row + 1 < grid_size {
                let rod = horizontal_rod_count + row * grid_size + column;
                let endpoint = 2 * rod;
                let residual = endpoint_or_residual[endpoint] - center;
                endpoint_or_residual[endpoint] = residual;
                residual_sum += residual;
            }

            hub_center_or_delta[hub_index] = residual_sum * hub_delta_scale[hub_index];
        }
    }
}

fn project_joint_constraints_direct(
    bodies: &mut [AffineBody],
    scratch: &mut SolverScratch,
    hub_count: usize,
    grid_size: usize,
) {
    debug_assert_eq!(hub_count, grid_size * grid_size);
    let SolverScratch {
        constraint_residual: endpoint_or_residual,
        hub_weighted_residual: hub_center_or_delta,
        hub_delta_scale,
        ..
    } = scratch;
    let (hubs, non_hubs) = bodies.split_at_mut(hub_count);
    let rod_count = endpoint_or_residual.len() / 2;
    let rods = &mut non_hubs[..rod_count];

    gather_joint_residuals_row_streaming(
        endpoint_or_residual,
        hub_center_or_delta,
        hub_delta_scale,
        grid_size,
    );

    let constraint_residual = endpoint_or_residual.as_slice();
    let hub_delta = hub_center_or_delta.as_slice();

    #[cfg(not(target_arch = "wasm32"))]
    if constraint_residual.len() >= PARALLEL_CORRECTION_JOINT_THRESHOLD
        && let Some(task_pool) = ComputeTaskPool::try_get()
    {
        let thread_count = task_pool.thread_num();
        if thread_count > 1 {
            let horizontal_rod_count = grid_size * (grid_size - 1);
            let hub_task_count = (thread_count / 3).max(1);
            let rod_task_count = thread_count.saturating_sub(hub_task_count).max(1);
            let hubs_per_task = hub_count.div_ceil(hub_task_count);
            let target_rods_per_task = rod_count.div_ceil(rod_task_count);
            let horizontal_rows_per_task = target_rods_per_task.div_ceil(grid_size - 1);
            let vertical_rows_per_task = target_rods_per_task.div_ceil(grid_size);
            let (horizontal_rods, vertical_rods) = rods.split_at_mut(horizontal_rod_count);
            let (horizontal_residuals, vertical_residuals) =
                constraint_residual.split_at(horizontal_rod_count * 2);
            task_pool.scope(|scope| {
                let horizontal_rods_per_task = horizontal_rows_per_task * (grid_size - 1);
                for (task_index, (rod_chunk, residual_chunk)) in horizontal_rods
                    .chunks_mut(horizontal_rods_per_task)
                    .zip(horizontal_residuals.chunks(horizontal_rods_per_task * 2))
                    .enumerate()
                {
                    let first_row = task_index * horizontal_rows_per_task;
                    scope.spawn(async move {
                        project_horizontal_rod_joint_constraints(
                            rod_chunk,
                            residual_chunk,
                            hub_delta,
                            grid_size,
                            first_row,
                        );
                    });
                }
                let vertical_rods_per_task = vertical_rows_per_task * grid_size;
                for (task_index, (rod_chunk, residual_chunk)) in vertical_rods
                    .chunks_mut(vertical_rods_per_task)
                    .zip(vertical_residuals.chunks(vertical_rods_per_task * 2))
                    .enumerate()
                {
                    let first_row = task_index * vertical_rows_per_task;
                    scope.spawn(async move {
                        project_vertical_rod_joint_constraints(
                            rod_chunk,
                            residual_chunk,
                            hub_delta,
                            grid_size,
                            first_row,
                        );
                    });
                }
                for (hub_chunk, delta_chunk) in hubs
                    .chunks_mut(hubs_per_task)
                    .zip(hub_delta.chunks(hubs_per_task))
                {
                    scope.spawn(async move {
                        apply_hub_joint_deltas(hub_chunk, delta_chunk);
                    });
                }
            });
            return;
        }
    }

    apply_hub_joint_deltas(hubs, hub_delta);
    project_rod_joint_constraints_structured(rods, constraint_residual, hub_delta, grid_size);
}

#[inline]
fn apply_hub_joint_deltas(hubs: &mut [AffineBody], hub_delta: &[DVec3]) {
    for (hub, &delta) in hubs.iter_mut().zip(hub_delta) {
        if !hub.fixed {
            for position in &mut hub.positions {
                *position += delta;
            }
        }
    }
}

#[inline]
fn apply_rod_joint_deltas(
    rod: &mut AffineBody,
    residual_pair: &[DVec3],
    start_hub_delta: DVec3,
    end_hub_delta: DVec3,
) {
    let start_correction = residual_pair[0] - start_hub_delta;
    let end_correction = residual_pair[1] - end_hub_delta;
    rod.positions[0] -= start_correction;
    rod.positions[1] -= start_correction;
    rod.positions[2] -= end_correction;
    rod.positions[3] -= end_correction;
}

#[inline]
fn project_horizontal_rod_joint_constraints(
    rods: &mut [AffineBody],
    residuals: &[DVec3],
    hub_delta: &[DVec3],
    grid_size: usize,
    first_row: usize,
) {
    let rods_per_row = grid_size - 1;
    debug_assert_eq!(rods.len() % rods_per_row, 0);
    debug_assert_eq!(residuals.len(), rods.len() * 2);

    for (row_offset, (rod_row, residual_row)) in rods
        .chunks_exact_mut(rods_per_row)
        .zip(residuals.chunks_exact(rods_per_row * 2))
        .enumerate()
    {
        let hub_start = (first_row + row_offset) * grid_size;
        let hub_row = &hub_delta[hub_start..hub_start + grid_size];
        for ((rod, residual_pair), hub_pair) in rod_row
            .iter_mut()
            .zip(residual_row.chunks_exact(2))
            .zip(hub_row.windows(2))
        {
            apply_rod_joint_deltas(rod, residual_pair, hub_pair[0], hub_pair[1]);
        }
    }
}

#[inline]
fn project_vertical_rod_joint_constraints(
    rods: &mut [AffineBody],
    residuals: &[DVec3],
    hub_delta: &[DVec3],
    grid_size: usize,
    first_row: usize,
) {
    debug_assert_eq!(rods.len() % grid_size, 0);
    debug_assert_eq!(residuals.len(), rods.len() * 2);

    for (row_offset, (rod_row, residual_row)) in rods
        .chunks_exact_mut(grid_size)
        .zip(residuals.chunks_exact(grid_size * 2))
        .enumerate()
    {
        let hub_start = (first_row + row_offset) * grid_size;
        let start_hub_row = &hub_delta[hub_start..hub_start + grid_size];
        let end_hub_row = &hub_delta[hub_start + grid_size..hub_start + 2 * grid_size];
        for (((rod, residual_pair), &start_hub_delta), &end_hub_delta) in rod_row
            .iter_mut()
            .zip(residual_row.chunks_exact(2))
            .zip(start_hub_row)
            .zip(end_hub_row)
        {
            apply_rod_joint_deltas(rod, residual_pair, start_hub_delta, end_hub_delta);
        }
    }
}

#[inline]
fn project_rod_joint_constraints_structured(
    rods: &mut [AffineBody],
    residuals: &[DVec3],
    hub_delta: &[DVec3],
    grid_size: usize,
) {
    let horizontal_rod_count = grid_size * (grid_size - 1);
    debug_assert_eq!(rods.len(), horizontal_rod_count * 2);
    debug_assert_eq!(residuals.len(), rods.len() * 2);
    let (horizontal_rods, vertical_rods) = rods.split_at_mut(horizontal_rod_count);
    let (horizontal_residuals, vertical_residuals) = residuals.split_at(horizontal_rod_count * 2);
    project_horizontal_rod_joint_constraints(
        horizontal_rods,
        horizontal_residuals,
        hub_delta,
        grid_size,
        0,
    );
    project_vertical_rod_joint_constraints(
        vertical_rods,
        vertical_residuals,
        hub_delta,
        grid_size,
        0,
    );
}

#[cfg(test)]
#[inline]
fn project_rod_joint_constraints_reference(
    rods: &mut [AffineBody],
    residuals: &[DVec3],
    joint_hubs: &[u32],
    hub_delta: &[DVec3],
) {
    for ((rod, residual_pair), hub_pair) in rods
        .iter_mut()
        .zip(residuals.chunks_exact(2))
        .zip(joint_hubs.chunks_exact(2))
    {
        apply_rod_joint_deltas(
            rod,
            residual_pair,
            hub_delta[hub_pair[0] as usize],
            hub_delta[hub_pair[1] as usize],
        );
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

#[cfg(test)]
fn apply_joint_correction(
    bodies: &mut [AffineBody],
    joints: &[BallJoint],
    multipliers: &[DVec3],
    hub_forces: &mut [DVec3],
    hub_count: usize,
    grid_size: usize,
) {
    debug_assert_eq!(hub_count, grid_size * grid_size);

    #[cfg(not(target_arch = "wasm32"))]
    if joints.len() >= PARALLEL_CORRECTION_JOINT_THRESHOLD
        && let Some(task_pool) = ComputeTaskPool::try_get()
    {
        let thread_count = task_pool.thread_num();
        if thread_count > 1 {
            let horizontal_rod_count = grid_size * (grid_size - 1);
            let hub_task_count = (thread_count / 3).max(1);
            let rod_task_count = thread_count.saturating_sub(hub_task_count).max(1);
            let hub_rows_per_task = grid_size.div_ceil(hub_task_count);
            let rod_count = joints.len() / 2;
            let rods_per_task = rod_count.div_ceil(rod_task_count);
            let (hubs, non_hubs) = bodies.split_at_mut(hub_count);
            let rods = &mut non_hubs[..rod_count];
            let hub_forces = &mut hub_forces[..hub_count];

            task_pool.scope(|scope| {
                for (task_index, (hub_chunk, force_chunk)) in hubs
                    .chunks_mut(hub_rows_per_task * grid_size)
                    .zip(hub_forces.chunks_mut(hub_rows_per_task * grid_size))
                    .enumerate()
                {
                    let first_row = task_index * hub_rows_per_task;
                    scope.spawn(async move {
                        for (row_offset, (hub_row, force_row)) in hub_chunk
                            .chunks_exact_mut(grid_size)
                            .zip(force_chunk.chunks_exact_mut(grid_size))
                            .enumerate()
                        {
                            let row = first_row + row_offset;
                            for (column, (hub, force_slot)) in
                                hub_row.iter_mut().zip(force_row).enumerate()
                            {
                                let mut force = DVec3::ZERO;
                                if column > 0 {
                                    let joint = 2 * (row * (grid_size - 1) + column - 1) + 1;
                                    force += -multipliers[joint] * HUB_CENTER[0];
                                }
                                if column + 1 < grid_size {
                                    let joint = 2 * (row * (grid_size - 1) + column);
                                    force += -multipliers[joint] * HUB_CENTER[0];
                                }
                                if row > 0 {
                                    let joint = 2
                                        * (horizontal_rod_count + (row - 1) * grid_size + column)
                                        + 1;
                                    force += -multipliers[joint] * HUB_CENTER[0];
                                }
                                if row + 1 < grid_size {
                                    let joint =
                                        2 * (horizontal_rod_count + row * grid_size + column);
                                    force += -multipliers[joint] * HUB_CENTER[0];
                                }
                                *force_slot = force;

                                if !hub.fixed {
                                    for position in &mut hub.positions {
                                        *position -= force * hub.inverse_diagonal;
                                    }
                                }
                            }
                        }
                    });
                }

                for (rod_chunk, multiplier_chunk) in rods
                    .chunks_mut(rods_per_task)
                    .zip(multipliers.chunks(rods_per_task * 2))
                {
                    scope.spawn(async move {
                        for (rod, multiplier_pair) in
                            rod_chunk.iter_mut().zip(multiplier_chunk.chunks_exact(2))
                        {
                            let start_correction =
                                (multiplier_pair[0] * 0.5) * rod.inverse_diagonal;
                            let end_correction = (multiplier_pair[1] * 0.5) * rod.inverse_diagonal;
                            rod.positions[0] -= start_correction;
                            rod.positions[1] -= start_correction;
                            rod.positions[2] -= end_correction;
                            rod.positions[3] -= end_correction;
                        }
                    });
                }
            });
            return;
        }
    }

    apply_joint_correction_sequential(
        bodies,
        joints,
        multipliers,
        hub_forces,
        hub_count,
        grid_size,
    );
}

#[cfg(test)]
fn apply_joint_correction_sequential(
    bodies: &mut [AffineBody],
    joints: &[BallJoint],
    multipliers: &[DVec3],
    hub_forces: &mut [DVec3],
    hub_count: usize,
    grid_size: usize,
) {
    debug_assert_eq!(hub_count, grid_size * grid_size);
    let horizontal_rod_count = grid_size * (grid_size - 1);

    for row in 0..grid_size {
        for column in 0..grid_size {
            let mut force = DVec3::ZERO;
            if column > 0 {
                let joint = 2 * (row * (grid_size - 1) + column - 1) + 1;
                force += -multipliers[joint] * HUB_CENTER[0];
            }
            if column + 1 < grid_size {
                let joint = 2 * (row * (grid_size - 1) + column);
                force += -multipliers[joint] * HUB_CENTER[0];
            }
            if row > 0 {
                let joint = 2 * (horizontal_rod_count + (row - 1) * grid_size + column) + 1;
                force += -multipliers[joint] * HUB_CENTER[0];
            }
            if row + 1 < grid_size {
                let joint = 2 * (horizontal_rod_count + row * grid_size + column);
                force += -multipliers[joint] * HUB_CENTER[0];
            }
            hub_forces[row * grid_size + column] = force;
        }
    }

    let (hubs, non_hubs) = bodies.split_at_mut(hub_count);
    let rods = &mut non_hubs[..joints.len() / 2];
    for (rod, multiplier_pair) in rods.iter_mut().zip(multipliers.chunks_exact(2)) {
        let start_correction = (multiplier_pair[0] * 0.5) * rod.inverse_diagonal;
        let end_correction = (multiplier_pair[1] * 0.5) * rod.inverse_diagonal;
        rod.positions[0] -= start_correction;
        rod.positions[1] -= start_correction;
        rod.positions[2] -= end_correction;
        rod.positions[3] -= end_correction;
    }

    for (hub, force) in hubs.iter_mut().zip(hub_forces) {
        if hub.fixed {
            continue;
        }
        for position in &mut hub.positions {
            *position -= *force * hub.inverse_diagonal;
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

#[cfg(test)]
fn body_rest_points(kind: BodyKind) -> [DVec3; 4] {
    match kind {
        BodyKind::Hub { .. } => hub_rest_points(),
        BodyKind::Rod => rod_rest_points(),
        BodyKind::Ball => ball_rest_points(),
    }
}

fn body_mass_per_point(kind: BodyKind) -> f64 {
    match kind {
        BodyKind::Hub { .. } => 0.18 / 4.0,
        BodyKind::Rod => 0.24 / 4.0,
        BodyKind::Ball => 1.2 / 4.0,
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

#[inline]
fn hub_attachment_center(points: &[DVec3; 4]) -> DVec3 {
    points[0] * 0.25 + points[1] * 0.25 + points[2] * 0.25 + points[3] * 0.25
}

#[inline]
fn rod_joint_endpoints(points: &[DVec3; 4]) -> [DVec3; 2] {
    [
        points[0] * 0.5 + points[1] * 0.5,
        points[2] * 0.5 + points[3] * 0.5,
    ]
}

#[inline]
fn rod_center_and_deformation_gradient(points: &[DVec3; 4]) -> (DVec3, DMat3) {
    let point_01 = points[0] + points[1];
    let point_23 = points[2] + points[3];
    let difference_01 = points[1] - points[0];
    let difference_23 = points[3] - points[2];
    let inverse_four_half_length = 1.0 / (2.0 * GRID_SPACING);
    let inverse_four_radius = 1.0 / (2.0 * ROD_THICKNESS as f64);

    (
        (point_01 + point_23) * 0.25,
        DMat3::from_cols(
            (point_23 - point_01) * inverse_four_half_length,
            (difference_01 + difference_23) * inverse_four_radius,
            (difference_01 - difference_23) * inverse_four_radius,
        ),
    )
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

#[derive(Clone, Copy)]
struct DVec3x2 {
    x: DVec2,
    y: DVec2,
    z: DVec2,
}

impl DVec3x2 {
    #[inline(always)]
    fn from_lanes(first: DVec3, second: DVec3) -> Self {
        Self {
            x: DVec2::new(first.x, second.x),
            y: DVec2::new(first.y, second.y),
            z: DVec2::new(first.z, second.z),
        }
    }

    #[inline(always)]
    fn lanes(self) -> (DVec3, DVec3) {
        (
            DVec3::new(self.x.x, self.y.x, self.z.x),
            DVec3::new(self.x.y, self.y.y, self.z.y),
        )
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }

    #[inline(always)]
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    #[inline(always)]
    fn scale(self, scale: DVec2) -> Self {
        Self {
            x: self.x * scale,
            y: self.y * scale,
            z: self.z * scale,
        }
    }

    #[inline(always)]
    fn dot(self, rhs: Self) -> DVec2 {
        (self.x * rhs.x) + (self.y * rhs.y) + (self.z * rhs.z)
    }

    #[inline(always)]
    fn cross(self, rhs: Self) -> Self {
        Self {
            x: self.y * rhs.z - rhs.y * self.z,
            y: self.z * rhs.x - rhs.z * self.x,
            z: self.x * rhs.y - rhs.x * self.y,
        }
    }
}

#[derive(Clone, Copy)]
struct DMat3x2 {
    x_axis: DVec3x2,
    y_axis: DVec3x2,
    z_axis: DVec3x2,
}

impl DMat3x2 {
    #[inline(always)]
    fn from_lanes(first: DMat3, second: DMat3) -> Self {
        Self {
            x_axis: DVec3x2::from_lanes(first.x_axis, second.x_axis),
            y_axis: DVec3x2::from_lanes(first.y_axis, second.y_axis),
            z_axis: DVec3x2::from_lanes(first.z_axis, second.z_axis),
        }
    }

    #[inline(always)]
    fn lanes(self) -> (DMat3, DMat3) {
        let (first_x, second_x) = self.x_axis.lanes();
        let (first_y, second_y) = self.y_axis.lanes();
        let (first_z, second_z) = self.z_axis.lanes();
        (
            DMat3::from_cols(first_x, first_y, first_z),
            DMat3::from_cols(second_x, second_y, second_z),
        )
    }

    #[inline(always)]
    fn determinant(self) -> DVec2 {
        self.z_axis.dot(self.x_axis.cross(self.y_axis))
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FinalPolarBranch {
    Skip,
    Positive,
    Negative,
}

#[inline(always)]
fn final_polar_branch(determinant: f64) -> Option<FinalPolarBranch> {
    if !determinant.is_finite() {
        None
    } else if determinant.abs() <= 1.0e-12 {
        Some(FinalPolarBranch::Skip)
    } else if determinant > 0.0 {
        Some(FinalPolarBranch::Positive)
    } else {
        Some(FinalPolarBranch::Negative)
    }
}

#[inline(always)]
fn normalize_pair(vector: DVec3x2) -> Result<Option<DVec3x2>, ()> {
    let length_squared = vector.dot(vector);
    let length = DVec2::new(length_squared.x.sqrt(), length_squared.y.sqrt());
    let reciprocal = DVec2::new(1.0 / length.x, 1.0 / length.y);
    let first_valid = reciprocal.x.is_finite() && reciprocal.x > 0.0;
    let second_valid = reciprocal.y.is_finite() && reciprocal.y > 0.0;
    match (first_valid, second_valid) {
        (true, true) => Ok(Some(vector.scale(reciprocal))),
        (false, false) => Ok(None),
        _ => Err(()),
    }
}

#[inline(always)]
fn orthonormalize_rotation_pair(rotation: DMat3x2) -> Option<DMat3x2> {
    let x = match normalize_pair(rotation.x_axis) {
        Ok(Some(x)) => x,
        Ok(None) => {
            return Some(DMat3x2::from_lanes(DMat3::IDENTITY, DMat3::IDENTITY));
        }
        Err(()) => return None,
    };
    let y_projection = x.scale(x.dot(rotation.y_axis));
    let mut y = match normalize_pair(rotation.y_axis.sub(y_projection)) {
        Ok(Some(y)) => y,
        Ok(None) => {
            return Some(DMat3x2::from_lanes(DMat3::IDENTITY, DMat3::IDENTITY));
        }
        Err(()) => return None,
    };

    let mut z = x.cross(y);
    let orientation = z.dot(rotation.z_axis);
    let first_flip = orientation.x < 0.0;
    let second_flip = orientation.y < 0.0;
    if first_flip != second_flip {
        return None;
    }
    if first_flip {
        z = z.neg();
    }
    y = z.cross(x);

    Some(DMat3x2 {
        x_axis: x,
        y_axis: y,
        z_axis: z,
    })
}

#[inline(always)]
fn closest_rotation_pair_2(first: DMat3, second: DMat3) -> Option<(DMat3, DMat3)> {
    let mut rotation = DMat3x2::from_lanes(first, second);
    let determinant = rotation.determinant();
    if !determinant.x.is_finite() || !determinant.y.is_finite() {
        return None;
    }
    let first_continue = determinant.x.abs() > 1.0e-12;
    let second_continue = determinant.y.abs() > 1.0e-12;
    if first_continue != second_continue {
        return None;
    }

    if first_continue {
        let cofactor_x = rotation.y_axis.cross(rotation.z_axis);
        let cofactor_y = rotation.z_axis.cross(rotation.x_axis);
        let cofactor_z = rotation.x_axis.cross(rotation.y_axis);
        let determinant = rotation.z_axis.dot(cofactor_z);
        let inverse_determinant = DVec2::new(1.0 / determinant.x, 1.0 / determinant.y);
        let half = DVec2::splat(0.5);
        rotation = DMat3x2 {
            x_axis: rotation
                .x_axis
                .add(cofactor_x.scale(inverse_determinant))
                .scale(half),
            y_axis: rotation
                .y_axis
                .add(cofactor_y.scale(inverse_determinant))
                .scale(half),
            z_axis: rotation
                .z_axis
                .add(cofactor_z.scale(inverse_determinant))
                .scale(half),
        };

        let cofactor_z = rotation.x_axis.cross(rotation.y_axis);
        let determinant = rotation.z_axis.dot(cofactor_z);
        let first_branch = final_polar_branch(determinant.x)?;
        let second_branch = final_polar_branch(determinant.y)?;
        if first_branch != second_branch {
            return None;
        }
        if first_branch != FinalPolarBranch::Skip {
            let cofactor_x = rotation.y_axis.cross(rotation.z_axis);
            let cofactor_y = rotation.z_axis.cross(rotation.x_axis);
            let scale = DVec2::new(determinant.x.abs(), determinant.y.abs());
            let scaled_x = rotation.x_axis.scale(scale);
            let scaled_y = rotation.y_axis.scale(scale);
            let scaled_z = rotation.z_axis.scale(scale);
            rotation = if first_branch == FinalPolarBranch::Positive {
                DMat3x2 {
                    x_axis: scaled_x.add(cofactor_x),
                    y_axis: scaled_y.add(cofactor_y),
                    z_axis: scaled_z.add(cofactor_z),
                }
            } else {
                DMat3x2 {
                    x_axis: scaled_x.sub(cofactor_x),
                    y_axis: scaled_y.sub(cofactor_y),
                    z_axis: scaled_z.sub(cofactor_z),
                }
            };
        }
    }

    orthonormalize_rotation_pair(rotation).map(DMat3x2::lanes)
}

fn closest_rotation(matrix: DMat3) -> DMat3 {
    closest_rotation_with_iterations::<POLAR_NEWTON_ITERATIONS>(matrix)
}

fn closest_rotation_with_iterations<const ITERATIONS: usize>(matrix: DMat3) -> DMat3 {
    let mut rotation = matrix;
    let mut final_round_pending = true;

    for _ in 1..ITERATIONS {
        if rotation.determinant().abs() <= 1.0e-12 {
            final_round_pending = false;
            break;
        }
        rotation = (rotation + rotation.inverse().transpose()) * 0.5;
    }

    if ITERATIONS > 0 && final_round_pending {
        let cofactor_z = rotation.x_axis.cross(rotation.y_axis);
        let determinant = rotation.z_axis.dot(cofactor_z);
        if determinant.abs() > 1.0e-12 {
            let cofactor_x = rotation.y_axis.cross(rotation.z_axis);
            let cofactor_y = rotation.z_axis.cross(rotation.x_axis);
            let scale = determinant.abs();
            rotation = if determinant > 0.0 {
                DMat3::from_cols(
                    rotation.x_axis * scale + cofactor_x,
                    rotation.y_axis * scale + cofactor_y,
                    rotation.z_axis * scale + cofactor_z,
                )
            } else {
                DMat3::from_cols(
                    rotation.x_axis * scale - cofactor_x,
                    rotation.y_axis * scale - cofactor_y,
                    rotation.z_axis * scale - cofactor_z,
                )
            };
        }
    }

    orthonormalize_rotation(rotation)
}

#[inline]
fn orthonormalize_rotation(rotation: DMat3) -> DMat3 {
    let Some(x) = rotation.x_axis.try_normalize() else {
        return DMat3::IDENTITY;
    };
    let Some(mut y) = (rotation.y_axis - x * x.dot(rotation.y_axis)).try_normalize() else {
        return DMat3::IDENTITY;
    };

    let mut z = x.cross(y);
    if z.dot(rotation.z_axis) < 0.0 {
        z = -z;
    }
    y = z.cross(x);

    DMat3::from_cols(x, y, z)
}

#[cfg(test)]
fn closest_rotation_with_divisive_final<const ITERATIONS: usize>(matrix: DMat3) -> DMat3 {
    let mut rotation = matrix;

    for _ in 0..ITERATIONS {
        if rotation.determinant().abs() <= 1.0e-12 {
            break;
        }
        rotation = (rotation + rotation.inverse().transpose()) * 0.5;
    }

    orthonormalize_rotation(rotation)
}

#[cfg(test)]
fn closest_rotation_with_legacy_tail<const ITERATIONS: usize>(matrix: DMat3) -> DMat3 {
    let mut rotation = matrix;

    for _ in 0..ITERATIONS {
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

    const STEP_TIME_GRID_ENV: &str = "STEP_TIME_GRID_SIZE";

    fn dvec3_bits(value: DVec3) -> [u64; 3] {
        [value.x.to_bits(), value.y.to_bits(), value.z.to_bits()]
    }

    fn assert_dvec3_slices_bit_exact(actual: &[DVec3], expected: &[DVec3], context: &str) {
        assert_eq!(actual.len(), expected.len(), "{context} length");
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                dvec3_bits(actual),
                dvec3_bits(expected),
                "{context} at vector {index}"
            );
        }
    }

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
    fn optimized_rotation_tail_matches_legacy_tail() {
        let mut maximum_component_error = 0.0_f64;
        let mut maximum_orthonormality_error = 0.0_f64;

        for scene in [
            DemoScene::JointGrid,
            DemoScene::CylinderDrape,
            DemoScene::FallingBalls,
        ] {
            let mut simulation = NetSimulation::new(scene);
            for step in 0..=150 {
                if matches!(step, 0 | 20 | 150) {
                    for body in &simulation.bodies {
                        if !matches!(body.kind, BodyKind::Rod) {
                            continue;
                        }
                        let gradient = rod_deformation_gradient(&body.positions);
                        let optimized = closest_rotation(gradient);
                        let legacy =
                            closest_rotation_with_legacy_tail::<POLAR_NEWTON_ITERATIONS>(gradient);
                        maximum_component_error = maximum_component_error.max(
                            optimized
                                .to_cols_array()
                                .into_iter()
                                .zip(legacy.to_cols_array())
                                .map(|(actual, expected)| (actual - expected).abs())
                                .fold(0.0_f64, f64::max),
                        );
                        let orthonormality = optimized.transpose() * optimized - DMat3::IDENTITY;
                        maximum_orthonormality_error = maximum_orthonormality_error.max(
                            orthonormality
                                .to_cols_array()
                                .into_iter()
                                .map(f64::abs)
                                .fold(0.0_f64, f64::max),
                        );
                    }
                }
                if step < 150 {
                    simulation.step(1.0 / DEFAULT_FIXED_HZ);
                }
            }
        }

        let zero = DMat3::from_cols(DVec3::ZERO, DVec3::ZERO, DVec3::ZERO);
        assert_eq!(closest_rotation(zero), DMat3::IDENTITY);
        assert_eq!(
            closest_rotation_with_legacy_tail::<POLAR_NEWTON_ITERATIONS>(zero),
            DMat3::IDENTITY
        );
        assert!(
            maximum_component_error < 2.0e-14,
            "optimized-vs-legacy rotation error: {maximum_component_error}"
        );
        assert!(
            maximum_orthonormality_error < 2.0e-14,
            "optimized rotation orthonormality error: {maximum_orthonormality_error}"
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn paired_polar_matches_scalar_bit_exactly() {
        fn assert_matrix_bits_eq(actual: DMat3, expected: DMat3, context: &str) {
            for (component, (actual, expected)) in actual
                .to_cols_array()
                .into_iter()
                .zip(expected.to_cols_array())
                .enumerate()
            {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "{context} component {component}"
                );
            }
        }

        let rotation = DMat3::from_quat(
            DQuat::from_rotation_x(0.37)
                * DQuat::from_rotation_y(-0.61)
                * DQuat::from_rotation_z(0.19),
        );
        let pairs = [
            (
                DMat3::IDENTITY,
                DMat3::from_diagonal(DVec3::new(0.35, 1.7, 3.2)),
            ),
            (
                rotation * DMat3::from_diagonal(DVec3::new(0.6, 1.4, 2.1)),
                DMat3::from_cols(
                    DVec3::new(1.0, 0.2, -0.1),
                    DVec3::new(0.35, 0.9, 0.15),
                    DVec3::new(-0.2, 0.1, 1.3),
                ),
            ),
            (
                DMat3::from_diagonal(DVec3::new(-1.0, 1.0, 1.0)),
                rotation * DMat3::from_diagonal(DVec3::new(-0.8, 1.2, 1.6)),
            ),
            (DMat3::ZERO, DMat3::ZERO),
            (
                DMat3::from_diagonal(DVec3::splat(1.0e-14)),
                DMat3::from_diagonal(DVec3::splat(1.0e-14)),
            ),
        ];

        for (index, (first, second)) in pairs.into_iter().enumerate() {
            let (paired_first, paired_second) = closest_rotation_pair_2(first, second)
                .unwrap_or_else(|| panic!("pair {index} unexpectedly required fallback"));
            assert_matrix_bits_eq(
                paired_first,
                closest_rotation_with_iterations::<2>(first),
                &format!("pair {index} first lane"),
            );
            assert_matrix_bits_eq(
                paired_second,
                closest_rotation_with_iterations::<2>(second),
                &format!("pair {index} second lane"),
            );

            let (swapped_second, swapped_first) = closest_rotation_pair_2(second, first)
                .unwrap_or_else(|| panic!("swapped pair {index} unexpectedly required fallback"));
            assert_matrix_bits_eq(
                swapped_first,
                paired_first,
                &format!("pair {index} lane swap"),
            );
            assert_matrix_bits_eq(
                swapped_second,
                paired_second,
                &format!("pair {index} lane swap"),
            );
        }

        assert!(closest_rotation_pair_2(DMat3::IDENTITY, DMat3::ZERO).is_none());
        assert!(
            closest_rotation_pair_2(
                DMat3::IDENTITY,
                DMat3::from_diagonal(DVec3::new(-1.0, 1.0, 1.0)),
            )
            .is_none()
        );
        let nan_matrix = DMat3::from_cols(DVec3::NAN, DVec3::NAN, DVec3::NAN);
        assert!(closest_rotation_pair_2(DMat3::IDENTITY, nan_matrix).is_none());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn paired_rod_shape_projection_matches_scalar_bit_exactly() {
        let mut simulation = NetSimulation::with_grid_size(DemoScene::JointGrid, 3);
        simulation.predict_for_test(1.0 / DEFAULT_FIXED_HZ);
        let hub_count = simulation.grid_size * simulation.grid_size;
        let mut paired = simulation.bodies[hub_count..hub_count + 3].to_vec();
        let collapsed = paired[1].positions[0];
        paired[1].positions = [collapsed; 4];
        let mut scalar = paired.clone();
        let mut paired_endpoints = vec![DVec3::ZERO; 6];
        let mut scalar_endpoints = vec![DVec3::ZERO; 6];

        project_rod_shapes::<2, true, true, true>(&mut paired, &mut paired_endpoints);
        project_rod_shapes::<2, true, true, false>(&mut scalar, &mut scalar_endpoints);

        for (body_index, (paired, scalar)) in paired.iter().zip(scalar).enumerate() {
            for point in 0..4 {
                assert_eq!(
                    dvec3_bits(paired.positions[point]),
                    dvec3_bits(scalar.positions[point]),
                    "body {body_index} point {point}"
                );
            }
        }
        assert_dvec3_slices_bit_exact(&paired_endpoints, &scalar_endpoints, "paired rod endpoints");
    }

    #[test]
    fn division_free_final_polar_round_matches_divisive_reference() {
        fn component_error(actual: DMat3, expected: DMat3) -> f64 {
            actual
                .to_cols_array()
                .into_iter()
                .zip(expected.to_cols_array())
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0_f64, f64::max)
        }

        fn compare_corpus<const ITERATIONS: usize>(matrices: &[DMat3]) -> f64 {
            matrices.iter().fold(0.0_f64, |maximum, &matrix| {
                maximum.max(component_error(
                    closest_rotation_with_iterations::<ITERATIONS>(matrix),
                    closest_rotation_with_divisive_final::<ITERATIONS>(matrix),
                ))
            })
        }

        let rotation = DMat3::from_quat(
            DQuat::from_rotation_x(0.37)
                * DQuat::from_rotation_y(-0.61)
                * DQuat::from_rotation_z(0.19),
        );
        let corpus = [
            DMat3::IDENTITY,
            DMat3::from_diagonal(DVec3::new(0.35, 1.7, 3.2)),
            rotation * DMat3::from_diagonal(DVec3::new(0.6, 1.4, 2.1)),
            DMat3::from_cols(
                DVec3::new(1.0, 0.2, -0.1),
                DVec3::new(0.35, 0.9, 0.15),
                DVec3::new(-0.2, 0.1, 1.3),
            ),
            DMat3::from_diagonal(DVec3::new(-1.0, 1.0, 1.0)),
            DMat3::ZERO,
            DMat3::from_diagonal(DVec3::new(1.0e-14, 1.0, 1.0)),
            DMat3::from_diagonal(DVec3::splat(1.0e100)),
        ];
        let corpus_error = compare_corpus::<1>(&corpus)
            .max(compare_corpus::<2>(&corpus))
            .max(compare_corpus::<3>(&corpus));

        for matrix in &corpus[5..] {
            assert_eq!(
                closest_rotation_with_iterations::<2>(*matrix),
                closest_rotation_with_divisive_final::<2>(*matrix)
            );
        }

        let mut maximum_live_error = 0.0_f64;
        let mut maximum_orthonormality_error = 0.0_f64;
        for scene in [
            DemoScene::JointGrid,
            DemoScene::CylinderDrape,
            DemoScene::FallingBalls,
        ] {
            let mut simulation = NetSimulation::new(scene);
            for step in 0..=150 {
                if matches!(step, 0 | 20 | 150) {
                    for body in &simulation.bodies {
                        if !matches!(body.kind, BodyKind::Rod) {
                            continue;
                        }
                        let gradient = rod_deformation_gradient(&body.positions);
                        let optimized = closest_rotation(gradient);
                        let divisive = closest_rotation_with_divisive_final::<
                            POLAR_NEWTON_ITERATIONS,
                        >(gradient);
                        maximum_live_error =
                            maximum_live_error.max(component_error(optimized, divisive));
                        maximum_orthonormality_error = maximum_orthonormality_error.max(
                            (optimized.transpose() * optimized - DMat3::IDENTITY)
                                .to_cols_array()
                                .into_iter()
                                .map(f64::abs)
                                .fold(0.0_f64, f64::max),
                        );
                    }
                }
                if step < 150 {
                    simulation.step(1.0 / DEFAULT_FIXED_HZ);
                }
            }
        }

        assert!(
            corpus_error < 1.0e-12,
            "division-free polar corpus error: {corpus_error}"
        );
        assert!(
            maximum_live_error < 5.0e-14,
            "division-free polar live error: {maximum_live_error}"
        );
        assert!(
            maximum_orthonormality_error < 2.0e-14,
            "division-free polar orthonormality error: {maximum_orthonormality_error}"
        );
    }

    #[test]
    fn division_free_final_polar_round_preserves_trajectory() {
        for scene in [
            DemoScene::JointGrid,
            DemoScene::CylinderDrape,
            DemoScene::FallingBalls,
        ] {
            let mut optimized = NetSimulation::new(scene);
            let mut reference = NetSimulation::new(scene);
            for _ in 0..150 {
                optimized.step(1.0 / DEFAULT_FIXED_HZ);
                reference.step_with_divisive_final_polar_round(1.0 / DEFAULT_FIXED_HZ);
            }

            let mut squared_error_sum = 0.0;
            let mut point_count = 0;
            let mut maximum_error = 0.0_f64;
            for (optimized_body, reference_body) in optimized.bodies.iter().zip(&reference.bodies) {
                for (optimized_point, reference_point) in optimized_body
                    .positions
                    .iter()
                    .zip(reference_body.positions)
                {
                    let error = (*optimized_point - reference_point).length();
                    squared_error_sum += error * error;
                    point_count += 1;
                    maximum_error = maximum_error.max(error);
                }
            }
            let rms_error = (squared_error_sum / point_count as f64).sqrt();
            assert!(simulation_is_finite(&optimized));
            assert!(simulation_is_finite(&reference));
            assert!(
                rms_error < 1.0e-9,
                "{} division-free polar RMS trajectory error: {rms_error}",
                scene.title()
            );
            assert!(
                maximum_error < 1.0e-8,
                "{} division-free polar maximum trajectory error: {maximum_error}",
                scene.title()
            );
        }
    }

    #[test]
    fn two_polar_iterations_match_higher_iteration_references() {
        fn reference_rotation(matrix: DMat3, iterations: usize) -> DMat3 {
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
            let mut maximum_two_iteration_error = 0.0_f64;
            let mut maximum_three_iteration_error = 0.0_f64;
            for _ in 0..150 {
                simulation.step(1.0 / DEFAULT_FIXED_HZ);
                for body in &simulation.bodies {
                    if !matches!(body.kind, BodyKind::Rod) {
                        continue;
                    }
                    let gradient = rod_deformation_gradient(&body.positions);
                    let three_iteration = reference_rotation(gradient, 3);
                    let five_iteration = reference_rotation(gradient, 5);
                    maximum_two_iteration_error = maximum_two_iteration_error.max(
                        closest_rotation(gradient)
                            .to_cols_array()
                            .into_iter()
                            .zip(three_iteration.to_cols_array())
                            .map(|(actual, expected)| (actual - expected).abs())
                            .fold(0.0_f64, f64::max),
                    );
                    maximum_three_iteration_error = maximum_three_iteration_error.max(
                        three_iteration
                            .to_cols_array()
                            .into_iter()
                            .zip(five_iteration.to_cols_array())
                            .map(|(actual, expected)| (actual - expected).abs())
                            .fold(0.0_f64, f64::max),
                    );
                }
            }
            assert!(
                maximum_two_iteration_error < 2.0e-6,
                "{} two-iteration polar error: {maximum_two_iteration_error}",
                scene.title()
            );
            assert!(
                maximum_three_iteration_error < 2.0e-9,
                "{} three-iteration polar error: {maximum_three_iteration_error}",
                scene.title()
            );
        }
    }

    #[test]
    fn two_polar_iterations_preserve_three_iteration_trajectory() {
        for scene in [
            DemoScene::JointGrid,
            DemoScene::CylinderDrape,
            DemoScene::FallingBalls,
        ] {
            let mut optimized = NetSimulation::new(scene);
            let mut reference = NetSimulation::new(scene);
            for _ in 0..150 {
                optimized.step(1.0 / DEFAULT_FIXED_HZ);
                reference.step_with_polar_iterations::<3>(1.0 / DEFAULT_FIXED_HZ);
            }

            let mut squared_error_sum = 0.0;
            let mut point_count = 0;
            let mut maximum_error = 0.0_f64;
            for (optimized_body, reference_body) in optimized.bodies.iter().zip(&reference.bodies) {
                for (optimized_point, reference_point) in optimized_body
                    .positions
                    .iter()
                    .zip(reference_body.positions)
                {
                    let error = (*optimized_point - reference_point).length();
                    squared_error_sum += error * error;
                    point_count += 1;
                    maximum_error = maximum_error.max(error);
                }
            }
            let rms_error = (squared_error_sum / point_count as f64).sqrt();
            assert!(simulation_is_finite(&optimized));
            assert!(simulation_is_finite(&reference));
            assert!(
                rms_error < 1.0e-5,
                "{} two-vs-three polar RMS trajectory error: {rms_error}",
                scene.title()
            );
            assert!(
                maximum_error < 5.0e-5,
                "{} two-vs-three polar maximum trajectory error: {maximum_error}",
                scene.title()
            );
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn parallel_shape_projection_matches_sequential_projection() {
        simulation_task_pool_options().create_default_pools();

        for scene in [DemoScene::JointGrid, DemoScene::FallingBalls] {
            let mut simulation = NetSimulation::with_grid_size(scene, 50);
            simulation.predict_for_test(1.0 / DEFAULT_FIXED_HZ);
            let mut sequential = simulation.bodies.clone();
            for body in &mut sequential {
                body.project_corotated_shape_with_polar_iterations::<POLAR_NEWTON_ITERATIONS>();
            }

            let hub_count = simulation.grid_size * simulation.grid_size;
            let rod_count = simulation.joints.len() / 2;
            let mut hub_centers = vec![DVec3::ZERO; hub_count];
            let mut rod_endpoints = vec![DVec3::ZERO; rod_count * 2];

            project_corotated_shapes::<POLAR_NEWTON_ITERATIONS, true, true>(
                &mut simulation.bodies,
                simulation.grid_size,
                &mut hub_centers,
                &mut rod_endpoints,
            );

            for (parallel, sequential) in simulation.bodies.iter().zip(sequential) {
                assert_eq!(parallel.positions, sequential.positions);
            }
            for (body, center) in simulation.bodies[..hub_count].iter().zip(hub_centers) {
                assert_eq!(center, hub_attachment_center(&body.positions));
            }
            for (body, endpoints) in simulation.bodies[hub_count..hub_count + rod_count]
                .iter()
                .zip(rod_endpoints.chunks_exact(2))
            {
                assert_eq!(endpoints, rod_joint_endpoints(&body.positions));
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn parallel_joint_residuals_match_sequential_projection() {
        simulation_task_pool_options().create_default_pools();

        let grid_size = GRID_SIZE_OPTIONS[3];
        let mut simulation = NetSimulation::with_grid_size(DemoScene::JointGrid, grid_size);
        simulation.predict_for_test(1.0 / DEFAULT_FIXED_HZ);
        let mut parallel = vec![DVec3::ZERO; simulation.joints.len()];
        let mut sequential = vec![DVec3::ZERO; simulation.joints.len()];
        let hub_count = grid_size * grid_size;

        compute_joint_residuals(
            &simulation.bodies,
            &simulation.joints,
            &mut parallel,
            hub_count,
        );
        let rods = &simulation.bodies[hub_count..hub_count + simulation.joints.len() / 2];
        for ((rod, joint_pair), residual_pair) in rods
            .iter()
            .zip(simulation.joints.chunks_exact(2))
            .zip(sequential.chunks_exact_mut(2))
        {
            let positions = &rod.positions;
            let start_hub = &simulation.bodies[joint_pair[0].b.body].positions;
            let end_hub = &simulation.bodies[joint_pair[1].b.body].positions;
            let start_hub_center = start_hub[0] * 0.25
                + start_hub[1] * 0.25
                + start_hub[2] * 0.25
                + start_hub[3] * 0.25;
            let end_hub_center =
                end_hub[0] * 0.25 + end_hub[1] * 0.25 + end_hub[2] * 0.25 + end_hub[3] * 0.25;
            residual_pair[0] = positions[0] * 0.5 + positions[1] * 0.5 - start_hub_center;
            residual_pair[1] = positions[2] * 0.5 + positions[3] * 0.5 - end_hub_center;
        }

        assert_eq!(parallel, sequential);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn parallel_joint_correction_matches_sequential_correction() {
        simulation_task_pool_options().create_default_pools();

        let grid_size = GRID_SIZE_OPTIONS[3];
        let hub_count = grid_size * grid_size;
        let mut simulation = NetSimulation::with_grid_size(DemoScene::JointGrid, grid_size);
        simulation.predict_for_test(1.0 / DEFAULT_FIXED_HZ);
        prepare_direct_joint_solver(
            &simulation.bodies,
            &simulation.joints,
            &mut simulation.solver_scratch,
        );
        compute_joint_residuals(
            &simulation.bodies,
            &simulation.joints,
            &mut simulation.solver_scratch.constraint_residual,
            hub_count,
        );
        solve_dual_direct(&mut simulation.solver_scratch, grid_size);

        let mut parallel = simulation.bodies.clone();
        let mut sequential = simulation.bodies;
        let mut parallel_hub_forces = vec![DVec3::ZERO; hub_count];
        let mut sequential_hub_forces = vec![DVec3::ZERO; hub_count];
        apply_joint_correction(
            &mut parallel,
            &simulation.joints,
            &simulation.solver_scratch.solution,
            &mut parallel_hub_forces,
            hub_count,
            grid_size,
        );
        apply_joint_correction_sequential(
            &mut sequential,
            &simulation.joints,
            &simulation.solver_scratch.solution,
            &mut sequential_hub_forces,
            hub_count,
            grid_size,
        );

        assert_eq!(parallel_hub_forces, sequential_hub_forces);
        for (parallel, sequential) in parallel.iter().zip(sequential) {
            assert_eq!(parallel.positions, sequential.positions);
        }
    }

    #[test]
    fn cached_joint_projection_matches_residual_projection() {
        #[cfg(not(target_arch = "wasm32"))]
        simulation_task_pool_options().create_default_pools();

        for grid_size in GRID_SIZE_OPTIONS {
            for scene in [
                DemoScene::JointGrid,
                DemoScene::CylinderDrape,
                DemoScene::FallingBalls,
            ] {
                let mut simulation = NetSimulation::with_grid_size(scene, grid_size);
                for _ in 0..20 {
                    simulation.step(1.0 / DEFAULT_FIXED_HZ);
                }

                let hub_count = grid_size * grid_size;
                let mut cached_bodies = simulation.bodies.clone();
                let mut residual_bodies = simulation.bodies.clone();
                let mut cached_scratch =
                    SolverScratch::new(cached_bodies.len(), &simulation.joints, hub_count);
                let mut residual_scratch =
                    SolverScratch::new(residual_bodies.len(), &simulation.joints, hub_count);
                project_corotated_shapes::<POLAR_NEWTON_ITERATIONS, true, true>(
                    &mut cached_bodies,
                    grid_size,
                    &mut cached_scratch.hub_weighted_residual,
                    &mut cached_scratch.constraint_residual,
                );
                project_corotated_shapes::<POLAR_NEWTON_ITERATIONS, true, true>(
                    &mut residual_bodies,
                    grid_size,
                    &mut residual_scratch.hub_weighted_residual,
                    &mut residual_scratch.constraint_residual,
                );
                assert_eq!(cached_bodies.len(), residual_bodies.len());
                assert_eq!(
                    cached_scratch.hub_weighted_residual,
                    residual_scratch.hub_weighted_residual
                );
                assert_eq!(
                    cached_scratch.constraint_residual,
                    residual_scratch.constraint_residual
                );

                prepare_direct_joint_solver(
                    &cached_bodies,
                    &simulation.joints,
                    &mut cached_scratch,
                );
                prepare_direct_joint_solver(
                    &residual_bodies,
                    &simulation.joints,
                    &mut residual_scratch,
                );
                let mut expected_residual = vec![DVec3::ZERO; simulation.joints.len()];
                compute_joint_residuals(
                    &residual_bodies,
                    &simulation.joints,
                    &mut expected_residual,
                    hub_count,
                );
                finalize_joint_residuals(
                    &mut residual_scratch.constraint_residual,
                    &residual_scratch.hub_weighted_residual,
                    &residual_scratch.joint_hub,
                );
                assert_eq!(residual_scratch.constraint_residual, expected_residual);

                project_joint_constraints_direct(
                    &mut cached_bodies,
                    &mut cached_scratch,
                    hub_count,
                    grid_size,
                );
                solve_dual_direct(&mut residual_scratch, grid_size);
                let expected_delta: Vec<DVec3> = residual_scratch
                    .hub_weighted_residual
                    .iter()
                    .zip(&residual_scratch.hub_schur_factor)
                    .map(|(&weighted_residual, &factor)| weighted_residual * factor)
                    .collect();
                assert_eq!(cached_scratch.constraint_residual, expected_residual);
                let maximum_delta_error = cached_scratch
                    .hub_weighted_residual
                    .iter()
                    .zip(&expected_delta)
                    .map(|(cached, expected)| (*cached - *expected).length())
                    .fold(0.0_f64, f64::max);
                assert!(
                    maximum_delta_error < 1.0e-12,
                    "{} {grid_size}x{grid_size} uniform-weight hub delta error: \
                     {maximum_delta_error}",
                    scene.title()
                );

                let rod_count = simulation.joints.len() / 2;
                let (residual_hubs, remaining) = residual_bodies.split_at_mut(hub_count);
                let residual_rods = &mut remaining[..rod_count];
                apply_hub_joint_deltas(residual_hubs, &expected_delta);
                project_rod_joint_constraints_reference(
                    residual_rods,
                    &expected_residual,
                    &residual_scratch.joint_hub,
                    &expected_delta,
                );
                let maximum_position_error = cached_bodies
                    .iter()
                    .zip(residual_bodies)
                    .flat_map(|(cached, residual)| {
                        cached
                            .positions
                            .iter()
                            .zip(residual.positions)
                            .map(|(cached, residual)| (*cached - residual).length())
                    })
                    .fold(0.0_f64, f64::max);
                assert!(
                    maximum_position_error < 1.0e-12,
                    "{} {grid_size}x{grid_size} uniform-weight projection error: \
                     {maximum_position_error}",
                    scene.title()
                );
            }
        }
    }

    #[test]
    fn direct_joint_projection_preserves_dual_trajectory() {
        for scene in [
            DemoScene::JointGrid,
            DemoScene::CylinderDrape,
            DemoScene::FallingBalls,
        ] {
            let mut direct = NetSimulation::new(scene);
            let mut dual = NetSimulation::new(scene);
            for _ in 0..150 {
                direct.step(1.0 / DEFAULT_FIXED_HZ);
                dual.step_with_dual_projection(1.0 / DEFAULT_FIXED_HZ);
            }

            let mut squared_error_sum = 0.0;
            let mut point_count = 0;
            let mut maximum_error = 0.0_f64;
            for (direct_body, dual_body) in direct.bodies.iter().zip(&dual.bodies) {
                for (direct_point, dual_point) in
                    direct_body.positions.iter().zip(dual_body.positions)
                {
                    let error = (*direct_point - dual_point).length();
                    squared_error_sum += error * error;
                    point_count += 1;
                    maximum_error = maximum_error.max(error);
                }
            }
            let rms_error = (squared_error_sum / point_count as f64).sqrt();
            println!(
                "DIRECT_PROJECTION_ERROR scene={} rms={rms_error:.3e} max={maximum_error:.3e}",
                scene.number()
            );
            assert!(
                rms_error < 1.0e-9,
                "{} direct-vs-dual RMS trajectory error: {rms_error}",
                scene.title()
            );
            assert!(
                maximum_error < 1.0e-8,
                "{} direct-vs-dual maximum trajectory error: {maximum_error}",
                scene.title()
            );
        }
    }

    fn report_step_time(scene: DemoScene) {
        #[cfg(not(target_arch = "wasm32"))]
        simulation_task_pool_options().create_default_pools();

        let dt = 1.0 / DEFAULT_FIXED_HZ;
        let grid_size = std::env::var(STEP_TIME_GRID_ENV)
            .ok()
            .map(|value| {
                value
                    .parse::<usize>()
                    .unwrap_or_else(|_| panic!("{STEP_TIME_GRID_ENV} must be an integer"))
            })
            .unwrap_or(DEFAULT_GRID_SIZE);
        assert!(grid_size >= 2, "benchmark grid must be at least 2x2");
        let (batches, warmup_steps, measured_steps) = match grid_size {
            2..=10 => (7, 20, 10),
            11..=25 => (5, 20, 5),
            26..=50 => (3, 20, 3),
            _ => (3, 20, 2),
        };
        let mut samples = Vec::with_capacity(batches);

        for _ in 0..batches {
            let mut simulation = NetSimulation::with_grid_size(scene, grid_size);
            for _ in 0..warmup_steps {
                simulation.step(dt);
            }

            let start = Instant::now();
            for _ in 0..measured_steps {
                simulation.step(dt);
            }
            let ms_per_step = start.elapsed().as_secs_f64() * 1_000.0 / measured_steps as f64;

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
            grid_size,
            batches,
            measured_steps,
            warmup_steps,
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
            simulation.predict_for_test(1.0 / DEFAULT_FIXED_HZ);
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

            solve_dual_direct(&mut simulation.solver_scratch, simulation.grid_size);

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
    fn row_streaming_joint_gather_matches_reference_bit_exactly() {
        for grid_size in [2, 3, 10, 25, 50, 100] {
            let endpoint_count = 4 * grid_size * (grid_size - 1);
            let hub_count = grid_size * grid_size;
            let endpoint = (0..endpoint_count)
                .map(|index| {
                    DVec3::new(
                        if index % 17 == 0 {
                            -0.0
                        } else {
                            ((index * 13) % 53) as f64 * 0.03125 - 0.75
                        },
                        ((index * 29) % 71) as f64 * 0.015625 - 0.5,
                        ((index * 43) % 89) as f64 * 0.0078125 - 0.25,
                    )
                })
                .collect::<Vec<_>>();
            let hub_center = (0..hub_count)
                .map(|index| {
                    DVec3::new(
                        ((index * 11) % 47) as f64 * 0.03125 - 0.5,
                        if index % 19 == 0 {
                            -0.0
                        } else {
                            ((index * 23) % 61) as f64 * 0.015625 - 0.375
                        },
                        ((index * 37) % 73) as f64 * 0.0078125 - 0.125,
                    )
                })
                .collect::<Vec<_>>();
            let hub_delta_scale = (0..hub_count)
                .map(|index| 0.125 + (index % 7) as f64 * 0.03125)
                .collect::<Vec<_>>();

            let mut row_streaming_endpoint = endpoint.clone();
            let mut reference_endpoint = endpoint;
            let mut row_streaming_hub = hub_center.clone();
            let mut reference_hub = hub_center;
            gather_joint_residuals_row_streaming(
                &mut row_streaming_endpoint,
                &mut row_streaming_hub,
                &hub_delta_scale,
                grid_size,
            );
            gather_joint_residuals_reference(
                &mut reference_endpoint,
                &mut reference_hub,
                &hub_delta_scale,
                grid_size,
            );
            assert_dvec3_slices_bit_exact(
                &row_streaming_endpoint,
                &reference_endpoint,
                &format!("{grid_size}x{grid_size} synthetic endpoint residual"),
            );
            assert_dvec3_slices_bit_exact(
                &row_streaming_hub,
                &reference_hub,
                &format!("{grid_size}x{grid_size} synthetic hub delta"),
            );
        }

        for scene in [
            DemoScene::JointGrid,
            DemoScene::CylinderDrape,
            DemoScene::FallingBalls,
        ] {
            let mut simulation = NetSimulation::with_grid_size(scene, 25);
            for _ in 0..8 {
                simulation.step(1.0 / DEFAULT_FIXED_HZ);
            }

            let hub_count = simulation.grid_size * simulation.grid_size;
            let mut projected_bodies = simulation.bodies.clone();
            for body in &mut projected_bodies {
                body.update_time_step_coefficients(1.0 / DEFAULT_FIXED_HZ);
            }
            let mut scratch =
                SolverScratch::new(projected_bodies.len(), &simulation.joints, hub_count);
            prepare_direct_joint_solver(&projected_bodies, &simulation.joints, &mut scratch);
            project_corotated_shapes::<POLAR_NEWTON_ITERATIONS, true, true>(
                &mut projected_bodies,
                simulation.grid_size,
                &mut scratch.hub_weighted_residual,
                &mut scratch.constraint_residual,
            );

            let mut row_streaming_endpoint = scratch.constraint_residual.clone();
            let mut reference_endpoint = scratch.constraint_residual;
            let mut row_streaming_hub = scratch.hub_weighted_residual.clone();
            let mut reference_hub = scratch.hub_weighted_residual;
            gather_joint_residuals_row_streaming(
                &mut row_streaming_endpoint,
                &mut row_streaming_hub,
                &scratch.hub_delta_scale,
                simulation.grid_size,
            );
            gather_joint_residuals_reference(
                &mut reference_endpoint,
                &mut reference_hub,
                &scratch.hub_delta_scale,
                simulation.grid_size,
            );
            assert_dvec3_slices_bit_exact(
                &row_streaming_endpoint,
                &reference_endpoint,
                &format!("{} live endpoint residual", scene.title()),
            );
            assert_dvec3_slices_bit_exact(
                &row_streaming_hub,
                &reference_hub,
                &format!("{} live hub delta", scene.title()),
            );
        }
    }

    #[test]
    fn structured_hub_gather_matches_joint_scatter() {
        let grid_size = GRID_SIZE_OPTIONS[3];
        let mut simulation = NetSimulation::with_grid_size(DemoScene::JointGrid, grid_size);
        simulation.predict_for_test(1.0 / DEFAULT_FIXED_HZ);
        prepare_direct_joint_solver(
            &simulation.bodies,
            &simulation.joints,
            &mut simulation.solver_scratch,
        );
        compute_joint_residuals(
            &simulation.bodies,
            &simulation.joints,
            &mut simulation.solver_scratch.constraint_residual,
            grid_size * grid_size,
        );

        let mut expected_hub_residual = vec![DVec3::ZERO; grid_size * grid_size];
        for (index, joint) in simulation.joints.iter().enumerate() {
            expected_hub_residual[joint.b.body] += simulation.solver_scratch.constraint_residual
                [index]
                * simulation.solver_scratch.rod_inverse_weight[index / 2];
        }
        let expected_solution = simulation
            .joints
            .iter()
            .enumerate()
            .map(|(index, joint)| {
                let hub = joint.b.body;
                (simulation.solver_scratch.constraint_residual[index]
                    - expected_hub_residual[hub] * simulation.solver_scratch.hub_schur_factor[hub])
                    * simulation.solver_scratch.rod_inverse_weight[index / 2]
            })
            .collect::<Vec<_>>();

        solve_dual_direct(&mut simulation.solver_scratch, grid_size);

        assert_eq!(
            simulation.solver_scratch.hub_weighted_residual,
            expected_hub_residual
        );
        assert_eq!(simulation.solver_scratch.solution, expected_solution);
    }

    #[test]
    fn structured_rod_correction_matches_joint_lookup_exactly() {
        for grid_size in [2, 10, 25, 50, 100] {
            for scene in [
                DemoScene::JointGrid,
                DemoScene::CylinderDrape,
                DemoScene::FallingBalls,
            ] {
                let simulation = NetSimulation::with_grid_size(scene, grid_size);
                let hub_count = grid_size * grid_size;
                let rod_count = simulation.joints.len() / 2;
                let rods = &simulation.bodies[hub_count..hub_count + rod_count];
                let residuals: Vec<_> = (0..rod_count * 2)
                    .map(|index| {
                        DVec3::new(
                            ((index * 17) % 31) as f64 - 15.0,
                            ((index * 29) % 37) as f64 - 18.0,
                            ((index * 43) % 47) as f64 - 23.0,
                        ) * 1.0e-5
                    })
                    .collect();
                let hub_delta: Vec<_> = (0..hub_count)
                    .map(|index| {
                        DVec3::new(
                            ((index * 13) % 19) as f64 - 9.0,
                            ((index * 23) % 29) as f64 - 14.0,
                            ((index * 31) % 41) as f64 - 20.0,
                        ) * 1.0e-6
                    })
                    .collect();

                let mut structured = rods.to_vec();
                let mut reference = rods.to_vec();
                project_rod_joint_constraints_structured(
                    &mut structured,
                    &residuals,
                    &hub_delta,
                    grid_size,
                );
                project_rod_joint_constraints_reference(
                    &mut reference,
                    &residuals,
                    &simulation.solver_scratch.joint_hub,
                    &hub_delta,
                );
                for (structured, reference) in structured.iter().zip(&reference) {
                    assert_eq!(structured.positions, reference.positions);
                }

                let horizontal_rod_count = grid_size * (grid_size - 1);
                let mut row_chunked = rods.to_vec();
                let (horizontal_rods, vertical_rods) =
                    row_chunked.split_at_mut(horizontal_rod_count);
                let (horizontal_residuals, vertical_residuals) =
                    residuals.split_at(horizontal_rod_count * 2);
                let horizontal_rows_per_chunk = 3;
                let horizontal_rods_per_chunk = horizontal_rows_per_chunk * (grid_size - 1);
                for (chunk, (rod_chunk, residual_chunk)) in horizontal_rods
                    .chunks_mut(horizontal_rods_per_chunk)
                    .zip(horizontal_residuals.chunks(horizontal_rods_per_chunk * 2))
                    .enumerate()
                {
                    project_horizontal_rod_joint_constraints(
                        rod_chunk,
                        residual_chunk,
                        &hub_delta,
                        grid_size,
                        chunk * horizontal_rows_per_chunk,
                    );
                }
                let vertical_rows_per_chunk = 4;
                let vertical_rods_per_chunk = vertical_rows_per_chunk * grid_size;
                for (chunk, (rod_chunk, residual_chunk)) in vertical_rods
                    .chunks_mut(vertical_rods_per_chunk)
                    .zip(vertical_residuals.chunks(vertical_rods_per_chunk * 2))
                    .enumerate()
                {
                    project_vertical_rod_joint_constraints(
                        rod_chunk,
                        residual_chunk,
                        &hub_delta,
                        grid_size,
                        chunk * vertical_rows_per_chunk,
                    );
                }
                for (chunked, reference) in row_chunked.iter().zip(&reference) {
                    assert_eq!(chunked.positions, reference.positions);
                }
            }
        }
    }

    #[test]
    fn production_rods_share_one_inverse_weight() {
        for grid_size in [2, 25, 50, 100] {
            for scene in [
                DemoScene::JointGrid,
                DemoScene::CylinderDrape,
                DemoScene::FallingBalls,
            ] {
                for dt in [1.0 / 30.0, 1.0 / 60.0, 1.0 / 120.0] {
                    let mut simulation = NetSimulation::with_grid_size(scene, grid_size);
                    for body in &mut simulation.bodies {
                        body.update_time_step_coefficients(dt);
                    }
                    prepare_direct_joint_solver(
                        &simulation.bodies,
                        &simulation.joints,
                        &mut simulation.solver_scratch,
                    );

                    let hub_count = grid_size * grid_size;
                    let rod_count = simulation.joints.len() / 2;
                    let first_rod_inverse_diagonal = simulation.bodies[hub_count].inverse_diagonal;
                    for rod in &simulation.bodies[hub_count..hub_count + rod_count] {
                        assert_eq!(
                            rod.inverse_diagonal.to_bits(),
                            first_rod_inverse_diagonal.to_bits()
                        );
                    }

                    let inverse_rod_weight = (first_rod_inverse_diagonal * 0.5).recip();
                    for (hub, &scale) in
                        simulation.solver_scratch.hub_delta_scale.iter().enumerate()
                    {
                        let expected =
                            inverse_rod_weight * simulation.solver_scratch.hub_schur_factor[hub];
                        assert_eq!(scale.to_bits(), expected.to_bits());
                    }
                }
            }
        }
    }

    #[test]
    fn per_rod_inverse_weights_match_per_joint_reference() {
        let grid_size = GRID_SIZE_OPTIONS[1];
        let hub_count = grid_size * grid_size;
        let mut simulation = NetSimulation::with_grid_size(DemoScene::JointGrid, grid_size);
        simulation.predict_for_test(1.0 / DEFAULT_FIXED_HZ);
        for (rod, body) in simulation.bodies[hub_count..].iter_mut().enumerate() {
            body.inverse_diagonal = 1.0e-5 + rod as f64 * 1.0e-10;
        }

        let mut expected_weights = Vec::with_capacity(simulation.joints.len());
        let mut expected_hub_sums = vec![0.0; hub_count];
        for joint in &simulation.joints {
            let inverse_rod_weight =
                (simulation.bodies[joint.a.body].inverse_diagonal * 0.5).recip();
            expected_weights.push(inverse_rod_weight);
            expected_hub_sums[joint.b.body] += inverse_rod_weight;
        }

        prepare_direct_joint_solver_reference(
            &simulation.bodies,
            &simulation.joints,
            &mut simulation.solver_scratch,
        );

        for (joint, expected) in expected_weights.iter().enumerate() {
            assert_eq!(
                simulation.solver_scratch.rod_inverse_weight[joint / 2].to_bits(),
                expected.to_bits()
            );
        }
        assert_eq!(
            simulation.solver_scratch.hub_inverse_rod_weight_sum,
            expected_hub_sums
        );
    }

    #[test]
    fn specialized_joint_correction_matches_generic_scatter() {
        for scene in [DemoScene::JointGrid, DemoScene::FallingBalls] {
            let mut simulation = NetSimulation::new(scene);
            simulation.predict_for_test(1.0 / DEFAULT_FIXED_HZ);

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

            let hub_count = simulation.grid_size * simulation.grid_size;
            let mut expected_hub_forces = vec![DVec3::ZERO; hub_count];
            for (joint, multiplier) in simulation.joints.iter().zip(&multipliers) {
                expected_hub_forces[joint.b.body] += -*multiplier * HUB_CENTER[0];
            }
            let mut hub_forces = vec![DVec3::ZERO; specialized.len()];
            apply_joint_correction(
                &mut specialized,
                &simulation.joints,
                &multipliers,
                &mut hub_forces,
                hub_count,
                simulation.grid_size,
            );
            apply_joint_correction_generic(&mut generic, &simulation.joints, &multipliers);
            assert_eq!(&hub_forces[..hub_count], expected_hub_forces);

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
    fn specialized_joint_residuals_match_generic_attachments() {
        for scene in [DemoScene::JointGrid, DemoScene::FallingBalls] {
            let mut simulation = NetSimulation::new(scene);
            for _ in 0..20 {
                simulation.step(1.0 / DEFAULT_FIXED_HZ);
            }

            let mut specialized = vec![DVec3::ZERO; simulation.joints.len()];
            let mut generic = vec![DVec3::ZERO; simulation.joints.len()];
            compute_joint_residuals(
                &simulation.bodies,
                &simulation.joints,
                &mut specialized,
                simulation.grid_size * simulation.grid_size,
            );
            compute_joint_residuals_generic(&simulation.bodies, &simulation.joints, &mut generic);

            let maximum_error = specialized
                .iter()
                .zip(&generic)
                .map(|(specialized, generic)| (*specialized - *generic).length())
                .fold(0.0_f64, f64::max);
            assert!(
                maximum_error < 1.0e-15,
                "{} specialized residual error: {maximum_error}",
                scene.title()
            );
        }
    }

    #[test]
    fn cached_ball_contacts_match_uncached_ordered_projection() {
        for warmup_steps in [0, 20, 100] {
            let mut simulation = NetSimulation::new(DemoScene::FallingBalls);
            for _ in 0..warmup_steps {
                simulation.step(1.0 / DEFAULT_FIXED_HZ);
            }

            let ball_indices = simulation.ball_indices.clone();
            let previous_positions = simulation.previous_positions;
            let mut cached = simulation.bodies.clone();
            let mut uncached = simulation.bodies;
            let chunk_count = cached.len().div_ceil(CONTACT_PROXY_CHUNK_SIZE);
            let mut chunk_min = vec![DVec3::ZERO; chunk_count];
            let mut chunk_max = vec![DVec3::ZERO; chunk_count];
            project_ball_contacts(
                &mut cached,
                &previous_positions,
                &ball_indices,
                &mut chunk_min,
                &mut chunk_max,
            );
            project_ball_contacts_uncached(&mut uncached, &previous_positions, &ball_indices);

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
                "cached contact projection error after {warmup_steps} steps: {maximum_error}"
            );
        }
    }

    #[test]
    fn specialized_contact_proxies_match_attachment_reference() {
        for (grid_size, warmup_steps) in [(10, 0), (10, 20), (10, 100), (25, 20)] {
            let mut simulation = NetSimulation::with_grid_size(DemoScene::FallingBalls, grid_size);
            for _ in 0..warmup_steps {
                simulation.step(1.0 / DEFAULT_FIXED_HZ);
            }

            let net_body_count = simulation.ball_indices[0];
            let mut maximum_error = 0.0_f64;
            let mut maximum_broad_bound_violation = 0.0_f64;
            for body_index in 0..net_body_count {
                let (proxy_start, proxy_end) =
                    detailed_contact_proxy(&simulation.bodies[body_index]);
                let (broad_start, broad_end) = broad_contact_proxy(&simulation.bodies[body_index]);
                let broad_minimum = broad_start.min(broad_end);
                let broad_maximum = broad_start.max(broad_end);
                for point in [proxy_start, proxy_end] {
                    for axis in 0..3 {
                        maximum_broad_bound_violation = maximum_broad_bound_violation
                            .max(broad_minimum[axis] - point[axis])
                            .max(point[axis] - broad_maximum[axis]);
                    }
                }
                let (expected_start, expected_end) = match simulation.bodies[body_index].kind {
                    BodyKind::Hub { .. } => {
                        let center = attachment_position(
                            &simulation.bodies,
                            Attachment {
                                body: body_index,
                                weights: HUB_CENTER,
                            },
                        );
                        (center, center)
                    }
                    BodyKind::Rod => {
                        let (start, end) = rod_collider_attachments(body_index);
                        (
                            attachment_position(&simulation.bodies, start),
                            attachment_position(&simulation.bodies, end),
                        )
                    }
                    BodyKind::Ball => unreachable!("balls follow all net bodies"),
                };
                maximum_error = maximum_error
                    .max((proxy_start - expected_start).length())
                    .max((proxy_end - expected_end).length());
            }
            assert!(
                maximum_error < 1.0e-13,
                "{grid_size}x{grid_size} proxy error after {warmup_steps} steps: {maximum_error}"
            );
            assert!(
                maximum_broad_bound_violation < 1.0e-13,
                "{grid_size}x{grid_size} broad bound violation after {warmup_steps} steps: \
                 {maximum_broad_bound_violation}"
            );
        }
    }

    #[test]
    fn on_demand_ball_contacts_match_detailed_bounds_exactly() {
        for (grid_size, warmup_steps) in [(10, 0), (10, 20), (10, 100), (25, 20), (25, 100)] {
            let mut simulation = NetSimulation::with_grid_size(DemoScene::FallingBalls, grid_size);
            for _ in 0..warmup_steps {
                simulation.step(1.0 / DEFAULT_FIXED_HZ);
            }

            let ball_indices = simulation.ball_indices.clone();
            let previous_positions = simulation.previous_positions;
            let mut on_demand = simulation.bodies.clone();
            let mut detailed_bounds = simulation.bodies;
            let chunk_count = on_demand.len().div_ceil(CONTACT_PROXY_CHUNK_SIZE);
            let mut on_demand_chunk_min = vec![DVec3::ZERO; chunk_count];
            let mut on_demand_chunk_max = vec![DVec3::ZERO; chunk_count];
            let mut detailed_chunk_min = vec![DVec3::ZERO; chunk_count];
            let mut detailed_chunk_max = vec![DVec3::ZERO; chunk_count];

            project_ball_contact_passes(
                &mut on_demand,
                &previous_positions,
                &ball_indices,
                &mut on_demand_chunk_min,
                &mut on_demand_chunk_max,
            );
            project_ball_contact_passes_detailed_bounds_reference(
                &mut detailed_bounds,
                &previous_positions,
                &ball_indices,
                &mut detailed_chunk_min,
                &mut detailed_chunk_max,
            );

            for (on_demand, detailed_bounds) in on_demand.iter().zip(&detailed_bounds) {
                assert_eq!(on_demand.positions, detailed_bounds.positions);
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn parallel_ball_contact_bounds_match_sequential_exactly() {
        simulation_task_pool_options().create_default_pools();
        let dt = 1.0 / DEFAULT_FIXED_HZ;
        for grid_size in [10, 25, 50, 100] {
            let mut simulation = NetSimulation::with_grid_size(DemoScene::FallingBalls, grid_size);

            for _ in 0..30 {
                simulation.step(dt);
            }

            let ball_indices = simulation.ball_indices.clone();
            let previous_positions = simulation.previous_positions;
            let chunk_count = ball_indices[0].div_ceil(CONTACT_PROXY_CHUNK_SIZE);
            let minimum_sentinel = DVec3::splat(12_345.0);
            let maximum_sentinel = DVec3::splat(-54_321.0);
            let mut parallel_min = vec![minimum_sentinel; chunk_count + 2];
            let mut parallel_max = vec![maximum_sentinel; chunk_count + 2];
            let mut sequential_min = vec![minimum_sentinel; chunk_count + 2];
            let mut sequential_max = vec![maximum_sentinel; chunk_count + 2];
            rebuild_ball_contact_chunk_bounds(
                &simulation.bodies,
                &ball_indices,
                &mut parallel_min,
                &mut parallel_max,
            );
            rebuild_ball_contact_chunk_bounds_reference(
                &simulation.bodies,
                &ball_indices,
                &mut sequential_min,
                &mut sequential_max,
            );
            assert_eq!(parallel_min, sequential_min);
            assert_eq!(parallel_max, sequential_max);
            assert_eq!(parallel_min[chunk_count..], [minimum_sentinel; 2]);
            assert_eq!(parallel_max[chunk_count..], [maximum_sentinel; 2]);

            let mut parallel_contacts = simulation.bodies.clone();
            let mut sequential_contacts = simulation.bodies;
            project_ball_pairs(&mut parallel_contacts, &previous_positions, &ball_indices);
            project_ball_pairs(&mut sequential_contacts, &previous_positions, &ball_indices);
            project_balls_against_chunked_net(
                &mut parallel_contacts,
                &previous_positions,
                &ball_indices,
                &mut parallel_min,
                &mut parallel_max,
            );
            project_balls_against_chunked_net(
                &mut sequential_contacts,
                &previous_positions,
                &ball_indices,
                &mut sequential_min,
                &mut sequential_max,
            );
            for (parallel_body, sequential_body) in
                parallel_contacts.iter().zip(&sequential_contacts)
            {
                assert_eq!(parallel_body.positions, sequential_body.positions);
            }
        }
    }

    #[test]
    fn reused_ball_contact_cache_matches_rebuild_each_pass() {
        for warmup_steps in [0, 20, 100] {
            let mut simulation = NetSimulation::new(DemoScene::FallingBalls);
            for _ in 0..warmup_steps {
                simulation.step(1.0 / DEFAULT_FIXED_HZ);
            }

            let ball_indices = simulation.ball_indices.clone();
            let previous_positions = simulation.previous_positions;
            let mut reused = simulation.bodies.clone();
            let mut rebuilt = simulation.bodies;
            let chunk_count = reused.len().div_ceil(CONTACT_PROXY_CHUNK_SIZE);
            let mut reused_chunk_min = vec![DVec3::ZERO; chunk_count];
            let mut reused_chunk_max = vec![DVec3::ZERO; chunk_count];
            let mut rebuilt_chunk_min = vec![DVec3::ZERO; chunk_count];
            let mut rebuilt_chunk_max = vec![DVec3::ZERO; chunk_count];

            project_ball_contact_passes(
                &mut reused,
                &previous_positions,
                &ball_indices,
                &mut reused_chunk_min,
                &mut reused_chunk_max,
            );
            for _ in 0..CONTACT_PASSES {
                project_ball_contacts(
                    &mut rebuilt,
                    &previous_positions,
                    &ball_indices,
                    &mut rebuilt_chunk_min,
                    &mut rebuilt_chunk_max,
                );
            }

            for (reused, rebuilt) in reused.iter().zip(&rebuilt) {
                assert_eq!(reused.positions, rebuilt.positions);
            }
        }
    }

    #[test]
    fn ball_contact_chunks_expand_after_moving_a_net_body() {
        let mut hub = AffineBody::new(
            BodyKind::Hub { fixed: false },
            hub_rest_points(),
            DVec3::ZERO,
            DQuat::IDENTITY,
            false,
        );
        hub.inverse_diagonal = 1.0;
        let mut left_ball = AffineBody::new(
            BodyKind::Ball,
            ball_rest_points(),
            DVec3::new(-0.270, 0.0, 0.0),
            DQuat::IDENTITY,
            false,
        );
        left_ball.inverse_diagonal = 1.0;
        let mut right_ball = AffineBody::new(
            BodyKind::Ball,
            ball_rest_points(),
            DVec3::new(0.416, 0.0, 0.0),
            DQuat::IDENTITY,
            false,
        );
        right_ball.inverse_diagonal = 1.0;

        let mut chunked = vec![hub, left_ball, right_ball];
        let mut uncached = chunked.clone();
        let previous_positions = chunked
            .iter()
            .map(|body| body.positions)
            .collect::<Vec<_>>();
        let initial_right_ball = chunked[2].centroid();
        let mut chunk_min = vec![DVec3::ZERO; 1];
        let mut chunk_max = vec![DVec3::ZERO; 1];

        project_ball_contacts(
            &mut chunked,
            &previous_positions,
            &[1, 2],
            &mut chunk_min,
            &mut chunk_max,
        );
        project_ball_contacts_uncached(&mut uncached, &previous_positions, &[1, 2]);

        let maximum_error = chunked
            .iter()
            .zip(&uncached)
            .flat_map(|(chunked, uncached)| {
                chunked
                    .positions
                    .iter()
                    .zip(&uncached.positions)
                    .map(|(chunked, uncached)| (*chunked - *uncached).length())
            })
            .fold(0.0_f64, f64::max);
        assert!(
            maximum_error < 1.0e-14,
            "chunk update error: {maximum_error}"
        );
        assert!(chunked[2].centroid().x > initial_right_ball.x);
        let (updated_start, updated_end) = detailed_contact_proxy(&chunked[0]);
        for point in [updated_start, updated_end] {
            assert!((0..3).all(|axis| {
                point[axis] >= chunk_min[0][axis] && point[axis] <= chunk_max[0][axis]
            }));
        }
    }

    #[test]
    fn specialized_cylinder_contacts_match_reference_projection() {
        let mut simulation = NetSimulation::new(DemoScene::CylinderDrape);
        for _ in 0..20 {
            simulation.step(1.0 / DEFAULT_FIXED_HZ);
        }

        let cylinder = simulation.cylinder.unwrap();
        let previous_positions = simulation.previous_positions;
        let mut specialized = simulation.bodies.clone();
        let mut reference = simulation.bodies;
        project_cylinder_contacts(&mut specialized, &previous_positions, cylinder);
        project_cylinder_contacts_reference(&mut reference, &previous_positions, cylinder);

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

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn body_major_cylinder_passes_match_pass_major_order_exactly() {
        simulation_task_pool_options().create_default_pools();

        for grid_size in GRID_SIZE_OPTIONS {
            let mut simulation = NetSimulation::with_grid_size(DemoScene::CylinderDrape, grid_size);
            for _ in 0..20 {
                simulation.step(1.0 / DEFAULT_FIXED_HZ);
            }
            let cylinder = simulation.cylinder.unwrap();
            let previous_positions = simulation.previous_positions;
            let mut body_major = simulation.bodies.clone();
            let mut pass_major = simulation.bodies;

            project_cylinder_contact_passes(&mut body_major, &previous_positions, cylinder);
            project_cylinder_contact_passes_pass_major_reference(
                &mut pass_major,
                &previous_positions,
                cylinder,
            );

            for (body_major, pass_major) in body_major.iter().zip(pass_major) {
                assert_eq!(body_major.positions, pass_major.positions);
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn body_major_cylinder_passes_preserve_full_trajectory_exactly() {
        simulation_task_pool_options().create_default_pools();
        let dt = 1.0 / DEFAULT_FIXED_HZ;

        for grid_size in GRID_SIZE_OPTIONS {
            let mut body_major = NetSimulation::with_grid_size(DemoScene::CylinderDrape, grid_size);
            let mut pass_major = NetSimulation::with_grid_size(DemoScene::CylinderDrape, grid_size);

            for step in 0..30 {
                body_major.step(dt);
                pass_major.step_with_pass_major_cylinder_contacts(dt);

                assert_eq!(
                    body_major.previous_positions, pass_major.previous_positions,
                    "previous-position mismatch for {grid_size}x{grid_size} at step {step}"
                );
                for (body_major, pass_major) in body_major.bodies.iter().zip(&pass_major.bodies) {
                    assert_eq!(
                        body_major.positions, pass_major.positions,
                        "position mismatch for {grid_size}x{grid_size} at step {step}"
                    );
                    assert_eq!(
                        body_major.predicted_positions, pass_major.predicted_positions,
                        "prediction mismatch for {grid_size}x{grid_size} at step {step}"
                    );
                }
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    #[ignore = "performance comparison for cylinder pass ordering"]
    fn compare_cylinder_pass_order_step_time() {
        simulation_task_pool_options().create_default_pools();
        let dt = 1.0 / DEFAULT_FIXED_HZ;

        for grid_size in [25, 50, 100] {
            let measured_steps = match grid_size {
                25 => 10,
                50 => 6,
                _ => 3,
            };
            let mut body_major = NetSimulation::with_grid_size(DemoScene::CylinderDrape, grid_size);
            let mut pass_major = NetSimulation::with_grid_size(DemoScene::CylinderDrape, grid_size);
            for _ in 0..20 {
                body_major.step(dt);
                pass_major.step_with_pass_major_cylinder_contacts(dt);
            }

            let mut body_major_samples = Vec::with_capacity(11);
            let mut pass_major_samples = Vec::with_capacity(11);
            for batch in 0..11 {
                let measure_body_major = |simulation: &mut NetSimulation| {
                    let start = Instant::now();
                    for _ in 0..measured_steps {
                        simulation.step(dt);
                    }
                    start.elapsed().as_secs_f64() * 1_000.0 / measured_steps as f64
                };
                let measure_pass_major = |simulation: &mut NetSimulation| {
                    let start = Instant::now();
                    for _ in 0..measured_steps {
                        simulation.step_with_pass_major_cylinder_contacts(dt);
                    }
                    start.elapsed().as_secs_f64() * 1_000.0 / measured_steps as f64
                };

                let (body_major_ms, pass_major_ms) = if batch % 2 == 0 {
                    (
                        measure_body_major(&mut body_major),
                        measure_pass_major(&mut pass_major),
                    )
                } else {
                    let pass_major_ms = measure_pass_major(&mut pass_major);
                    let body_major_ms = measure_body_major(&mut body_major);
                    (body_major_ms, pass_major_ms)
                };
                body_major_samples.push(body_major_ms);
                pass_major_samples.push(pass_major_ms);
                assert_eq!(body_major.previous_positions, pass_major.previous_positions);
                for (body_major, pass_major) in body_major.bodies.iter().zip(&pass_major.bodies) {
                    assert_eq!(body_major.positions, pass_major.positions);
                }
            }

            body_major_samples.sort_by(f64::total_cmp);
            pass_major_samples.sort_by(f64::total_cmp);
            let body_major_ms = body_major_samples[body_major_samples.len() / 2];
            let pass_major_ms = pass_major_samples[pass_major_samples.len() / 2];
            let change_percent = (body_major_ms / pass_major_ms - 1.0) * 100.0;
            println!(
                "CYLINDER_PASS_TIME grid={grid_size} body_major_ms={body_major_ms:.4} \
                 pass_major_ms={pass_major_ms:.4} change_percent={change_percent:.2}"
            );
        }
    }

    #[test]
    fn weighted_shape_projection_matches_reference() {
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
                let mut specialized = body.clone();
                let mut reference = body.clone();
                specialized.project_corotated_shape();
                reference.project_corotated_shape_reference();
                for (specialized, reference) in
                    specialized.positions.iter().zip(reference.positions)
                {
                    maximum_error = maximum_error.max((*specialized - reference).length());
                }
            }
            assert!(
                maximum_error < 2.0e-15,
                "{} weighted shape projection error: {maximum_error}",
                scene.title()
            );
        }
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
            let mut maximum_displacement_spread = 0.0_f64;
            for (body, previous_positions) in
                simulation.bodies.iter().zip(&simulation.previous_positions)
            {
                if !matches!(body.kind, BodyKind::Hub { .. } | BodyKind::Ball) {
                    continue;
                }

                let center = body.centroid();
                let rest_points = body_rest_points(body.kind);
                for index in 0..4 {
                    maximum_shape_error = maximum_shape_error
                        .max((body.positions[index] - center - rest_points[index]).length());
                    let displacement = body.positions[index] - previous_positions[index];
                    let first_displacement = body.positions[0] - previous_positions[0];
                    maximum_displacement_spread = maximum_displacement_spread
                        .max((displacement - first_displacement).length());
                }
            }

            assert!(
                maximum_shape_error < 1.0e-10,
                "{} center-attached shape error: {maximum_shape_error}",
                scene.title()
            );
            assert!(
                maximum_displacement_spread < 1.0e-10,
                "{} center-attached displacement spread: {maximum_displacement_spread}",
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
                let rest_points = rod_rest_points();
                let covariance = (0..4).fold(DMat3::ZERO, |sum, index| {
                    let position = body.positions[index] - center;
                    let rest = rest_points[index];
                    sum + DMat3::from_cols(position * rest.x, position * rest.y, position * rest.z)
                });
                let rest_covariance = rest_points.iter().fold(DMat3::ZERO, |sum, rest| {
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
    fn fused_rod_geometry_preserves_legacy_trajectory() {
        let mut maximum_center_error = 0.0_f64;
        let mut maximum_gradient_error = 0.0_f64;
        for grid_size in [10, 25, 50, 100] {
            for scene in [
                DemoScene::JointGrid,
                DemoScene::CylinderDrape,
                DemoScene::FallingBalls,
            ] {
                let mut simulation = NetSimulation::with_grid_size(scene, grid_size);
                for step in 0..=20 {
                    if matches!(step, 0 | 20) {
                        let hub_count = grid_size * grid_size;
                        let rod_count = simulation.joints.len() / 2;
                        for rod in &simulation.bodies[hub_count..hub_count + rod_count] {
                            let (center, gradient) =
                                rod_center_and_deformation_gradient(&rod.positions);
                            maximum_center_error = maximum_center_error
                                .max((center - centroid(&rod.positions)).length());
                            maximum_gradient_error = maximum_gradient_error.max(
                                gradient
                                    .to_cols_array()
                                    .into_iter()
                                    .zip(rod_deformation_gradient(&rod.positions).to_cols_array())
                                    .map(|(fused, legacy)| (fused - legacy).abs())
                                    .fold(0.0_f64, f64::max),
                            );
                        }
                    }
                    if step < 20 {
                        simulation.step(1.0 / DEFAULT_FIXED_HZ);
                    }
                }
            }
        }
        assert!(
            maximum_center_error < 5.0e-14,
            "fused rod center error: {maximum_center_error}"
        );
        assert!(
            maximum_gradient_error < 5.0e-14,
            "fused rod gradient error: {maximum_gradient_error}"
        );

        for scene in [
            DemoScene::JointGrid,
            DemoScene::CylinderDrape,
            DemoScene::FallingBalls,
        ] {
            let mut fused = NetSimulation::new(scene);
            let mut legacy = NetSimulation::new(scene);
            for _ in 0..150 {
                fused.step(1.0 / DEFAULT_FIXED_HZ);
                legacy.step_with_legacy_rod_geometry(1.0 / DEFAULT_FIXED_HZ);
            }

            let mut squared_error_sum = 0.0;
            let mut point_count = 0;
            let mut maximum_error = 0.0_f64;
            for (fused_body, legacy_body) in fused.bodies.iter().zip(&legacy.bodies) {
                for (fused_point, legacy_point) in
                    fused_body.positions.iter().zip(legacy_body.positions)
                {
                    let error = (*fused_point - legacy_point).length();
                    squared_error_sum += error * error;
                    point_count += 1;
                    maximum_error = maximum_error.max(error);
                }
            }
            let rms_error = (squared_error_sum / point_count as f64).sqrt();
            println!(
                "FUSED_ROD_GEOMETRY_ERROR scene={} rms={rms_error:.3e} max={maximum_error:.3e}",
                scene.number()
            );
            assert!(
                rms_error < 1.0e-9,
                "{} fused rod geometry RMS trajectory error: {rms_error}",
                scene.title()
            );
            assert!(
                maximum_error < 1.0e-8,
                "{} fused rod geometry maximum trajectory error: {maximum_error}",
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
                let rest_points = body_rest_points(body.kind);
                (0..4).map(move |index| {
                    (body.positions[index] - center - rotation * rest_points[index]).length()
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
    fn previous_position_history_is_kept_out_of_hot_body_state() {
        assert_eq!(std::mem::size_of::<AffineBody>(), 216);
        for scene in [
            DemoScene::JointGrid,
            DemoScene::CylinderDrape,
            DemoScene::FallingBalls,
        ] {
            let simulation = NetSimulation::new(scene);
            assert_eq!(simulation.bodies.len(), simulation.previous_positions.len());
            for (body, previous_positions) in
                simulation.bodies.iter().zip(&simulation.previous_positions)
            {
                assert_eq!(body.positions, *previous_positions);
            }
        }
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
    fn deferred_velocity_reconstruction_matches_explicit_recurrence() {
        for fixed in [false, true] {
            let mut deferred = AffineBody::new(
                BodyKind::Rod,
                rod_rest_points(),
                DVec3::new(0.3, 1.7, -0.4),
                DQuat::from_rotation_z(0.37),
                fixed,
            );
            let mut explicit = deferred.clone();
            let mut deferred_previous_positions = deferred.positions;
            let mut explicit_previous_positions;
            let mut explicit_velocity = [DVec3::ZERO; 4];
            let mut previous_velocity_scale = 0.0;

            for (step, dt) in [1.0 / 30.0, 1.0 / 30.0, 1.0 / 60.0, 1.0 / 120.0, 1.0 / 30.0]
                .into_iter()
                .enumerate()
            {
                deferred.predict_positions(
                    &mut deferred_previous_positions,
                    dt,
                    previous_velocity_scale,
                );

                explicit_previous_positions = explicit.positions;
                if fixed {
                    explicit.predicted_positions = explicit.positions;
                    explicit_velocity = [DVec3::ZERO; 4];
                } else {
                    for (index, velocity) in explicit_velocity.iter().enumerate() {
                        explicit.predicted_positions[index] =
                            explicit.positions[index] + *velocity * dt + GRAVITY * (dt * dt);
                        explicit.positions[index] = explicit.predicted_positions[index];
                    }
                }

                assert_eq!(deferred.positions, explicit.positions);
                assert_eq!(deferred_previous_positions, explicit_previous_positions);
                assert_eq!(deferred.predicted_positions, explicit.predicted_positions);

                if !fixed {
                    for index in 0..4 {
                        let correction = DVec3::new(
                            (step + index) as f64 * 1.0e-4,
                            (2 * step + index) as f64 * -2.0e-4,
                            (step + 3 * index) as f64 * 1.5e-4,
                        );
                        deferred.positions[index] += correction;
                        explicit.positions[index] += correction;
                    }
                }

                previous_velocity_scale = velocity_damping_for_dt(dt) / dt;
                if !fixed {
                    for (index, velocity) in explicit_velocity.iter_mut().enumerate() {
                        *velocity = (explicit.positions[index]
                            - explicit_previous_positions[index])
                            * previous_velocity_scale;
                    }
                }
            }
        }
    }

    #[test]
    fn cached_time_step_coefficients_match_forced_rebuilds() {
        for scene in [
            DemoScene::JointGrid,
            DemoScene::CylinderDrape,
            DemoScene::FallingBalls,
        ] {
            let mut cached = NetSimulation::new(scene);
            let mut rebuilt = NetSimulation::new(scene);
            for dt in [1.0 / 30.0, 1.0 / 30.0, 1.0 / 60.0, 1.0 / 60.0, 1.0 / 30.0] {
                cached.step(dt);
                rebuilt.coefficient_dt_bits = u64::MAX;
                rebuilt.step(dt);

                for (cached_body, rebuilt_body) in cached.bodies.iter().zip(&rebuilt.bodies) {
                    assert_eq!(cached_body.positions, rebuilt_body.positions);
                    assert_eq!(
                        cached_body.predicted_positions,
                        rebuilt_body.predicted_positions
                    );
                    assert_eq!(
                        cached_body.inertia.to_bits(),
                        rebuilt_body.inertia.to_bits()
                    );
                    assert_eq!(
                        cached_body.inverse_diagonal.to_bits(),
                        rebuilt_body.inverse_diagonal.to_bits()
                    );
                }
                assert_eq!(cached.previous_positions, rebuilt.previous_positions);
                assert_eq!(
                    cached.solver_scratch.rod_inverse_weight,
                    rebuilt.solver_scratch.rod_inverse_weight
                );
                assert_eq!(
                    cached.solver_scratch.hub_inverse_rod_weight_sum,
                    rebuilt.solver_scratch.hub_inverse_rod_weight_sum
                );
                assert_eq!(
                    cached.solver_scratch.hub_schur_factor,
                    rebuilt.solver_scratch.hub_schur_factor
                );
                assert_eq!(
                    cached.solver_scratch.hub_delta_scale,
                    rebuilt.solver_scratch.hub_delta_scale
                );
                assert_eq!(
                    cached.previous_velocity_scale.to_bits(),
                    rebuilt.previous_velocity_scale.to_bits()
                );
            }
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
        let previous_positions = bodies.iter().map(|body| body.positions).collect::<Vec<_>>();

        project_attachment_pair(
            &mut bodies,
            &previous_positions,
            first,
            second,
            1.0,
            DVec3::X,
        );

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
        let previous_positions = bodies.iter().map(|body| body.positions).collect::<Vec<_>>();

        project_cylinder_contacts(&mut bodies, &previous_positions, cylinder);

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
                false,
            );
            rod.inverse_diagonal = 1.0;
            let mut ball = AffineBody::new(
                BodyKind::Ball,
                ball_rest_points(),
                sphere_position,
                DQuat::IDENTITY,
                false,
            );
            ball.inverse_diagonal = 1.0;
            let mut bodies = vec![rod, ball];
            let previous_positions = bodies.iter().map(|body| body.positions).collect::<Vec<_>>();

            let mut chunk_min = vec![DVec3::ZERO; 1];
            let mut chunk_max = vec![DVec3::ZERO; 1];
            project_ball_contacts(
                &mut bodies,
                &previous_positions,
                &[1],
                &mut chunk_min,
                &mut chunk_max,
            );

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
                    false,
                );
                body.inverse_diagonal = 1.0;
                body
            })
            .collect::<Vec<_>>();
        let previous_positions = bodies.iter().map(|body| body.positions).collect::<Vec<_>>();

        let mut chunk_min = vec![DVec3::ZERO; 1];
        let mut chunk_max = vec![DVec3::ZERO; 1];
        project_ball_contacts(
            &mut bodies,
            &previous_positions,
            &[0, 1],
            &mut chunk_min,
            &mut chunk_max,
        );

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
        simulation.bodies.len() == simulation.previous_positions.len()
            && simulation
                .bodies
                .iter()
                .zip(&simulation.previous_positions)
                .all(|(body, previous_positions)| {
                    body.positions
                        .iter()
                        .chain(previous_positions)
                        .chain(&body.predicted_positions)
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

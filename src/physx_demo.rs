use bevy::prelude::{Quat, Transform};

#[cfg(not(target_arch = "wasm32"))]
use bevy::prelude::Vec3;

#[cfg(not(target_arch = "wasm32"))]
mod platform {
    use super::*;
    use crate::DemoScene;
    use physx::prelude::*;
    use physx::traits::Class;
    use std::ptr::NonNull;

    const GRID_SIZE: usize = 10;
    const GRID_SPACING: f32 = 0.55;
    const HUB_RADIUS: f32 = 0.075;
    const ROD_THICKNESS: f32 = 0.055;
    const ROD_LENGTH_FACTOR: f32 = 0.78;
    const BALL_RADIUS: f32 = 0.34;
    const CYLINDER_RADIUS: f32 = 0.70;
    const CYLINDER_LENGTH: f32 = 5.80;

    const HUB_MASS: f32 = 0.18;
    const ROD_MASS: f32 = 0.24;
    const BALL_MASS: f32 = 1.20;

    const NET_LAYER: CollisionLayers = CollisionLayers::Ghost;
    const COLLIDER_LAYER: CollisionLayers = CollisionLayers::Terrain;
    const BALL_LAYER: CollisionLayers = CollisionLayers::Static;

    type PxMaterial = physx::material::PxMaterial<()>;
    type PxShape = physx::shape::PxShape<(), PxMaterial>;
    type PxArticulationLink = physx::articulation_link::PxArticulationLink<usize, PxShape>;
    type PxRigidStatic = physx::rigid_static::PxRigidStatic<usize, PxShape>;
    type PxRigidDynamic = physx::rigid_dynamic::PxRigidDynamic<usize, PxShape>;
    type PxArticulationReducedCoordinate =
        physx::articulation_reduced_coordinate::PxArticulationReducedCoordinate<
            usize,
            PxArticulationLink,
        >;
    type PxScene = physx::scene::PxScene<
        usize,
        PxArticulationLink,
        PxRigidStatic,
        PxRigidDynamic,
        PxArticulationReducedCoordinate,
        NoopCallbacks,
        NoopCallbacks,
        NoopCallbacks,
        NoopCallbacks,
        NoopCallbacks,
    >;

    struct NoopCallbacks;

    impl CollisionCallback for NoopCallbacks {
        fn on_collision(
            &mut self,
            _: &physx_sys::PxContactPairHeader,
            _: &[physx_sys::PxContactPair],
        ) {
        }
    }

    impl TriggerCallback for NoopCallbacks {
        fn on_trigger(&mut self, _: &[physx_sys::PxTriggerPair]) {}
    }

    impl ConstraintBreakCallback for NoopCallbacks {
        fn on_constraint_break(&mut self, _: &[physx_sys::PxConstraintInfo]) {}
    }

    impl WakeSleepCallback<PxArticulationLink, PxRigidStatic, PxRigidDynamic> for NoopCallbacks {
        fn on_wake_sleep(
            &mut self,
            _: &[&physx::actor::ActorMap<PxArticulationLink, PxRigidStatic, PxRigidDynamic>],
            _: bool,
        ) {
        }
    }

    impl AdvanceCallback<PxArticulationLink, PxRigidDynamic> for NoopCallbacks {
        fn on_advance(
            &self,
            _: &[&physx::rigid_body::RigidBodyMap<PxArticulationLink, PxRigidDynamic>],
            _: &[PxTransform],
        ) {
        }
    }

    struct SphericalJoint(NonNull<physx_sys::PxSphericalJoint>);

    impl Drop for SphericalJoint {
        fn drop(&mut self) {
            // PxJoint is the owning base class of PxSphericalJoint. Releasing the
            // joint also removes its constraint from the scene.
            unsafe {
                physx_sys::PxJoint_release_mut(self.0.as_ptr().cast::<physx_sys::PxJoint>());
            }
        }
    }

    struct PhysxWorld {
        // Fields drop in declaration order: joints before their scene, then the
        // scene before the material that its actors' shapes reference.
        _joints: Vec<SphericalJoint>,
        scene: Owner<PxScene>,
        _material: Owner<PxMaterial>,
        body_actors: Vec<*mut physx_sys::PxRigidActor>,
        body_transforms: Vec<Transform>,
    }

    pub struct PhysxDemo {
        // The world must be released before the foundation/physics instance.
        world: Option<PhysxWorld>,
        physics: physx::physics::PhysicsFoundation<physx::foundation::DefaultAllocator, PxShape>,
    }

    #[derive(Clone, Copy)]
    struct RodConnection {
        rod: usize,
        start_hub: usize,
        end_hub: usize,
    }

    impl PhysxDemo {
        pub fn new() -> Self {
            let mut builder = physx::physics::PhysicsFoundationBuilder::default();
            builder.with_extensions(true);
            let physics = builder
                .build::<PxShape>()
                .expect("PhysX should initialize with extensions");

            Self {
                world: None,
                physics,
            }
        }

        pub fn reset(&mut self, scene: DemoScene) {
            self.world = None;
            self.world = Some(Self::create_world(&mut self.physics, scene));
            self.refresh_body_transforms();
        }

        fn create_world(
            physics: &mut physx::physics::PhysicsFoundation<
                physx::foundation::DefaultAllocator,
                PxShape,
            >,
            demo_scene: DemoScene,
        ) -> PhysxWorld {
            let mut scene: Owner<PxScene> = physics
                .create(SceneDescriptor {
                    gravity: PxVec3::new(0.0, -9.81, 0.0),
                    solver_type: SolverType::Tgs,
                    simulation_filter_shader: FilterShaderDescriptor::CallDefaultFirst(
                        simulation_filter,
                    ),
                    ..SceneDescriptor::new(0usize)
                })
                .expect("PhysX should create the demo scene");
            let mut material = physics
                .create_material(0.25, 0.20, 0.03, ())
                .expect("PhysX should create the demo material");

            let mut body_actors = Vec::with_capacity(if demo_scene == DemoScene::FallingBalls {
                283
            } else {
                280
            });
            let mut initial_transforms = Vec::with_capacity(body_actors.capacity());
            let mut hub_indices = [[0usize; GRID_SIZE]; GRID_SIZE];
            let mut hub_positions = [[Vec3::ZERO; GRID_SIZE]; GRID_SIZE];
            let mut connections = Vec::with_capacity(180);
            let grid_height = if demo_scene == DemoScene::FallingBalls {
                1.70
            } else {
                3.0
            };

            let hub_geometry = PxSphereGeometry::new(HUB_RADIUS);
            let hub_density = HUB_MASS / sphere_volume(HUB_RADIUS);

            for row in 0..GRID_SIZE {
                for column in 0..GRID_SIZE {
                    let position = Vec3::new(
                        (column as f32 - (GRID_SIZE - 1) as f32 * 0.5) * GRID_SPACING,
                        grid_height,
                        if demo_scene == DemoScene::FallingBalls {
                            (row as f32 - (GRID_SIZE - 1) as f32 * 0.5) * GRID_SPACING
                        } else {
                            -(row as f32) * GRID_SPACING
                        },
                    );
                    let fixed = match demo_scene {
                        DemoScene::JointGrid | DemoScene::CylinderDrape => row == 0,
                        DemoScene::FallingBalls => {
                            (row == 0 || row == GRID_SIZE - 1)
                                && (column == 0 || column == GRID_SIZE - 1)
                        }
                    };
                    let body_index = body_actors.len();
                    let pose = px_transform(position, Quat::IDENTITY);
                    let actor_ptr = if fixed {
                        let mut actor = physics
                            .create_rigid_static(
                                pose,
                                &hub_geometry,
                                material.as_mut(),
                                PxTransform::default(),
                                body_index,
                            )
                            .expect("PhysX should create a fixed hub");
                        actor.set_collision_filter(NET_LAYER, COLLIDER_LAYER | BALL_LAYER, 0, 0);
                        let ptr = actor.as_mut_ptr();
                        scene.add_static_actor(actor);
                        ptr
                    } else {
                        let mut actor = physics
                            .create_rigid_dynamic(
                                pose,
                                &hub_geometry,
                                material.as_mut(),
                                hub_density,
                                PxTransform::default(),
                                body_index,
                            )
                            .expect("PhysX should create a moving hub");
                        configure_dynamic(&mut actor);
                        actor.set_collision_filter(NET_LAYER, COLLIDER_LAYER | BALL_LAYER, 0, 0);
                        let ptr = actor.as_mut_ptr();
                        scene.add_dynamic_actor(actor);
                        ptr
                    };

                    hub_indices[row][column] = body_index;
                    hub_positions[row][column] = position;
                    body_actors.push(actor_ptr);
                    initial_transforms
                        .push(Transform::from_translation(position).with_rotation(Quat::IDENTITY));
                }
            }

            let rod_geometry = PxBoxGeometry::new(
                GRID_SPACING * ROD_LENGTH_FACTOR * 0.5,
                ROD_THICKNESS * 0.5,
                ROD_THICKNESS * 0.5,
            );
            let rod_density =
                ROD_MASS / (GRID_SPACING * ROD_LENGTH_FACTOR * ROD_THICKNESS * ROD_THICKNESS);

            for row in 0..GRID_SIZE {
                for column in 0..(GRID_SIZE - 1) {
                    add_rod(
                        physics,
                        &mut scene,
                        &mut material,
                        &rod_geometry,
                        rod_density,
                        hub_indices[row][column],
                        hub_indices[row][column + 1],
                        hub_positions[row][column],
                        hub_positions[row][column + 1],
                        &mut body_actors,
                        &mut initial_transforms,
                        &mut connections,
                    );
                }
            }

            for row in 0..(GRID_SIZE - 1) {
                for column in 0..GRID_SIZE {
                    add_rod(
                        physics,
                        &mut scene,
                        &mut material,
                        &rod_geometry,
                        rod_density,
                        hub_indices[row][column],
                        hub_indices[row + 1][column],
                        hub_positions[row][column],
                        hub_positions[row + 1][column],
                        &mut body_actors,
                        &mut initial_transforms,
                        &mut connections,
                    );
                }
            }

            debug_assert_eq!(body_actors.len(), 280);
            debug_assert_eq!(connections.len(), 180);

            if demo_scene == DemoScene::CylinderDrape {
                // PhysX capsules are X-axis aligned, matching this scene's cylinder.
                // PxCapsuleGeometry's length is 2 * (half-height + radius).
                let cylinder_geometry = PxCapsuleGeometry::new(
                    CYLINDER_RADIUS,
                    CYLINDER_LENGTH * 0.5 - CYLINDER_RADIUS,
                );
                let mut cylinder = physics
                    .create_rigid_static(
                        PxTransform::from_translation(&PxVec3::new(0.0, 1.75, -2.30)),
                        &cylinder_geometry,
                        material.as_mut(),
                        PxTransform::default(),
                        usize::MAX,
                    )
                    .expect("PhysX should create the fixed cylinder");
                cylinder.set_collision_filter(COLLIDER_LAYER, NET_LAYER, 0, 0);
                scene.add_static_actor(cylinder);
            }

            if demo_scene == DemoScene::FallingBalls {
                let ball_geometry = PxSphereGeometry::new(BALL_RADIUS);
                let ball_density = BALL_MASS / sphere_volume(BALL_RADIUS);

                for position in [
                    Vec3::new(-1.10, 2.80, -0.80),
                    Vec3::new(0.85, 3.15, -0.25),
                    Vec3::new(-0.20, 3.50, 1.05),
                ] {
                    let body_index = body_actors.len();
                    let mut actor = physics
                        .create_rigid_dynamic(
                            px_transform(position, Quat::IDENTITY),
                            &ball_geometry,
                            material.as_mut(),
                            ball_density,
                            PxTransform::default(),
                            body_index,
                        )
                        .expect("PhysX should create a falling ball");
                    configure_dynamic(&mut actor);
                    actor.set_collision_filter(BALL_LAYER, NET_LAYER | BALL_LAYER, 0, 0);
                    let ptr = actor.as_mut_ptr();
                    scene.add_dynamic_actor(actor);
                    body_actors.push(ptr);
                    initial_transforms.push(Transform::from_translation(position));
                }
            }

            let physics_ptr: *mut physx_sys::PxPhysics = physics.as_mut_ptr();
            let mut joints = Vec::with_capacity(360);
            let hub_frame = PxTransform::default();
            let start_frame =
                PxTransform::from_translation(&PxVec3::new(-GRID_SPACING * 0.5, 0.0, 0.0));
            let end_frame =
                PxTransform::from_translation(&PxVec3::new(GRID_SPACING * 0.5, 0.0, 0.0));

            for connection in connections {
                joints.push(create_spherical_joint(
                    physics_ptr,
                    body_actors[connection.rod],
                    &start_frame,
                    body_actors[connection.start_hub],
                    &hub_frame,
                ));
                joints.push(create_spherical_joint(
                    physics_ptr,
                    body_actors[connection.rod],
                    &end_frame,
                    body_actors[connection.end_hub],
                    &hub_frame,
                ));
            }

            debug_assert_eq!(joints.len(), 360);

            PhysxWorld {
                _joints: joints,
                scene,
                _material: material,
                body_actors,
                body_transforms: initial_transforms,
            }
        }

        pub fn step(&mut self, dt: f32) {
            let Some(world) = &mut self.world else {
                return;
            };

            world
                .scene
                .step(
                    dt,
                    None::<&mut physx_sys::PxBaseTask>,
                    None::<&mut ScratchBuffer>,
                    true,
                )
                .expect("PhysX simulation step should succeed");
        }

        pub fn refresh_body_transforms(&mut self) {
            let Some(world) = &mut self.world else {
                return;
            };

            for (body_index, actor) in world.body_actors.iter().copied().enumerate() {
                let raw_pose = unsafe { physx_sys::PxRigidActor_getGlobalPose(actor) };
                let pose: PxTransform = raw_pose.into();
                let position = pose.translation();
                let rotation = pose.rotation();
                world.body_transforms[body_index] =
                    Transform::from_xyz(position.x(), position.y(), position.z()).with_rotation(
                        Quat::from_xyzw(rotation.x(), rotation.y(), rotation.z(), rotation.w())
                            .normalize(),
                    );
            }
        }

        pub fn body_transforms(&self) -> &[Transform] {
            self.world
                .as_ref()
                .map_or(&[], |world| world.body_transforms.as_slice())
        }

        pub fn status(&self) -> &'static str {
            "READY"
        }
    }

    fn add_rod(
        physics: &mut physx::physics::PhysicsFoundation<
            physx::foundation::DefaultAllocator,
            PxShape,
        >,
        scene: &mut Owner<PxScene>,
        material: &mut Owner<PxMaterial>,
        geometry: &PxBoxGeometry,
        density: f32,
        start_hub: usize,
        end_hub: usize,
        start: Vec3,
        end: Vec3,
        body_actors: &mut Vec<*mut physx_sys::PxRigidActor>,
        body_transforms: &mut Vec<Transform>,
        connections: &mut Vec<RodConnection>,
    ) {
        let direction = (end - start).normalize();
        let rotation = Quat::from_rotation_arc(Vec3::X, direction);
        let position = (start + end) * 0.5;
        let body_index = body_actors.len();
        let mut actor = physics
            .create_rigid_dynamic(
                px_transform(position, rotation),
                geometry,
                material.as_mut(),
                density,
                PxTransform::default(),
                body_index,
            )
            .expect("PhysX should create a rod");
        configure_dynamic(&mut actor);
        actor.set_collision_filter(NET_LAYER, COLLIDER_LAYER | BALL_LAYER, 0, 0);
        let ptr = actor.as_mut_ptr();
        scene.add_dynamic_actor(actor);

        body_actors.push(ptr);
        body_transforms.push(Transform::from_translation(position).with_rotation(rotation));
        connections.push(RodConnection {
            rod: body_index,
            start_hub,
            end_hub,
        });
    }

    fn configure_dynamic(actor: &mut Owner<PxRigidDynamic>) {
        actor.set_linear_damping(0.09);
        actor.set_angular_damping(0.09);
        actor.set_solver_iteration_counts(16, 4);
    }

    fn create_spherical_joint(
        physics: *mut physx_sys::PxPhysics,
        actor_a: *mut physx_sys::PxRigidActor,
        frame_a: &PxTransform,
        actor_b: *mut physx_sys::PxRigidActor,
        frame_b: &PxTransform,
    ) -> SphericalJoint {
        let joint = unsafe {
            physx_sys::phys_PxSphericalJointCreate(
                physics,
                actor_a,
                frame_a.as_ptr(),
                actor_b,
                frame_b.as_ptr(),
            )
        };
        SphericalJoint(NonNull::new(joint).expect("PhysX should create a spherical joint"))
    }

    fn px_transform(position: Vec3, rotation: Quat) -> PxTransform {
        PxTransform::from_translation_rotation(
            &PxVec3::new(position.x, position.y, position.z),
            &PxQuat::new(rotation.x, rotation.y, rotation.z, rotation.w),
        )
    }

    fn sphere_volume(radius: f32) -> f32 {
        4.0 / 3.0 * std::f32::consts::PI * radius.powi(3)
    }

    unsafe extern "C" fn simulation_filter(
        callback: *mut physx_sys::FilterShaderCallbackInfo,
    ) -> physx_sys::PxFilterFlags {
        let callback = unsafe { &*callback };
        let a_accepts_b = callback.filterData0.word0 & callback.filterData1.word1 != 0;
        let b_accepts_a = callback.filterData1.word0 & callback.filterData0.word1 != 0;

        if a_accepts_b && b_accepts_a {
            physx_sys::PxFilterFlags::default()
        } else {
            physx_sys::PxFilterFlags::Suppress
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn all_scenes_have_stable_body_indexed_poses() {
            let mut demo = PhysxDemo::new();

            for (scene, expected_bodies) in [
                (DemoScene::JointGrid, 280),
                (DemoScene::CylinderDrape, 280),
                (DemoScene::FallingBalls, 283),
            ] {
                demo.reset(scene);
                assert_eq!(demo.body_transforms().len(), expected_bodies);
                assert!(
                    demo.body_transforms()
                        .iter()
                        .all(|pose| pose.translation.is_finite())
                );
                for _ in 0..120 {
                    demo.step(1.0 / 120.0);
                }
                demo.refresh_body_transforms();
                assert!(
                    demo.body_transforms()
                        .iter()
                        .all(|pose| pose.translation.is_finite())
                );
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod platform {
    use super::*;
    use crate::DemoScene;
    use js_sys::Float32Array;
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen(module = "/src/physx_bridge.js")]
    extern "C" {
        fn physx_begin_load();
        fn physx_reset(scene_id: u32);
        fn physx_step(dt: f32);
        fn physx_refresh_transforms();
        fn physx_transforms() -> Float32Array;
        fn physx_status_code() -> u32;
    }

    pub struct PhysxDemo {
        body_transforms: Vec<Transform>,
        packed_transforms: Vec<f32>,
    }

    impl PhysxDemo {
        pub fn new() -> Self {
            physx_begin_load();
            Self {
                body_transforms: Vec::new(),
                packed_transforms: Vec::new(),
            }
        }

        pub fn reset(&mut self, scene: DemoScene) {
            self.body_transforms.clear();
            self.packed_transforms.clear();
            physx_reset(match scene {
                DemoScene::JointGrid => 1,
                DemoScene::CylinderDrape => 2,
                DemoScene::FallingBalls => 3,
            });
            self.refresh_body_transforms();
        }

        pub fn step(&mut self, dt: f32) {
            physx_step(dt);
        }

        pub fn refresh_body_transforms(&mut self) {
            if physx_status_code() != 1 {
                return;
            }

            physx_refresh_transforms();
            let packed = physx_transforms();
            let packed_len = packed.length() as usize;
            if packed_len == 0 || !packed_len.is_multiple_of(7) {
                return;
            }

            self.packed_transforms.resize(packed_len, 0.0);
            packed.copy_to(&mut self.packed_transforms);
            self.body_transforms.clear();
            self.body_transforms
                .reserve_exact(self.packed_transforms.len() / 7);

            for pose in self.packed_transforms.chunks_exact(7) {
                let rotation = Quat::from_xyzw(pose[3], pose[4], pose[5], pose[6]);
                let rotation = if rotation.is_finite() && rotation.length_squared() > 1.0e-12 {
                    rotation.normalize()
                } else {
                    Quat::IDENTITY
                };
                self.body_transforms
                    .push(Transform::from_xyz(pose[0], pose[1], pose[2]).with_rotation(rotation));
            }
        }

        pub fn body_transforms(&self) -> &[Transform] {
            &self.body_transforms
        }

        pub fn status(&self) -> &'static str {
            match physx_status_code() {
                1 => "READY",
                2 => "ERROR",
                _ => "LOADING",
            }
        }
    }
}

pub use platform::PhysxDemo;

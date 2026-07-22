use bevy::prelude::{Quat, Transform, Vec3};

const STARTING_CUBE_POSITION: Vec3 = Vec3::new(0.0, 3.0, 0.0);

#[cfg(not(target_arch = "wasm32"))]
mod platform {
    use super::*;
    use physx::prelude::*;

    type PxMaterial = physx::material::PxMaterial<()>;
    type PxShape = physx::shape::PxShape<(), PxMaterial>;
    type PxArticulationLink = physx::articulation_link::PxArticulationLink<(), PxShape>;
    type PxRigidStatic = physx::rigid_static::PxRigidStatic<(), PxShape>;
    type PxRigidDynamic = physx::rigid_dynamic::PxRigidDynamic<(), PxShape>;
    type PxArticulationReducedCoordinate =
        physx::articulation_reduced_coordinate::PxArticulationReducedCoordinate<
            (),
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

    pub struct PhysxDemo {
        // Struct fields drop in declaration order. The scene and material must be
        // released before the foundation and physics objects that created them.
        scene: Owner<PxScene>,
        _material: Owner<PxMaterial>,
        _physics: physx::physics::PhysicsFoundation<physx::foundation::DefaultAllocator, PxShape>,
        cube_transform: Transform,
    }

    impl PhysxDemo {
        pub fn new() -> Self {
            let mut physics = PhysicsFoundation::<_, PxShape>::default();
            let (scene, material) = Self::create_world(&mut physics);

            let mut demo = Self {
                scene,
                _material: material,
                _physics: physics,
                cube_transform: Transform::from_translation(STARTING_CUBE_POSITION),
            };
            demo.refresh_cube_transform();
            demo
        }

        pub fn reset(&mut self) {
            let (scene, material) = Self::create_world(&mut self._physics);
            self.scene = scene;
            self._material = material;
            self.refresh_cube_transform();
        }

        fn create_world(
            physics: &mut physx::physics::PhysicsFoundation<
                physx::foundation::DefaultAllocator,
                PxShape,
            >,
        ) -> (Owner<PxScene>, Owner<PxMaterial>) {
            let mut scene: Owner<PxScene> = physics
                .create(SceneDescriptor {
                    gravity: PxVec3::new(0.0, -9.81, 0.0),
                    // physx 0.19 reads small scene user data through a pointer-sized
                    // slot, so use an aligned pointer-sized value instead of `()`.
                    ..SceneDescriptor::new(0usize)
                })
                .expect("PhysX should create the demo scene");
            let mut material = physics
                .create_material(0.55, 0.45, 0.05, ())
                .expect("PhysX should create the demo material");

            let ground_geometry = PxBoxGeometry::new(3.0, 0.5, 3.0);
            let ground = physics
                .create_rigid_static(
                    PxTransform::from_translation(&PxVec3::new(0.0, -0.5, 0.0)),
                    &ground_geometry,
                    material.as_mut(),
                    PxTransform::default(),
                    (),
                )
                .expect("PhysX should create the fixed cube");
            scene.add_static_actor(ground);

            let cube_geometry = PxBoxGeometry::new(0.5, 0.5, 0.5);
            let cube = physics
                .create_rigid_dynamic(
                    PxTransform::from_translation(&PxVec3::new(
                        STARTING_CUBE_POSITION.x,
                        STARTING_CUBE_POSITION.y,
                        STARTING_CUBE_POSITION.z,
                    )),
                    &cube_geometry,
                    material.as_mut(),
                    1.0,
                    PxTransform::default(),
                    (),
                )
                .expect("PhysX should create the falling cube");
            scene.add_dynamic_actor(cube);

            (scene, material)
        }

        pub fn step(&mut self, dt: f32) {
            self.scene
                .step(
                    dt,
                    None::<&mut physx_sys::PxBaseTask>,
                    None::<&mut ScratchBuffer>,
                    true,
                )
                .expect("PhysX simulation step should succeed");
            self.refresh_cube_transform();
        }

        pub fn cube_transform(&self) -> Transform {
            self.cube_transform
        }

        pub fn status(&self) -> &'static str {
            "READY"
        }

        fn refresh_cube_transform(&mut self) {
            let actors = self.scene.get_dynamic_actors();
            let pose = actors[0].get_global_pose();
            let position = pose.translation();
            let rotation = pose.rotation();
            self.cube_transform = Transform::from_xyz(position.x(), position.y(), position.z())
                .with_rotation(
                    Quat::from_xyzw(rotation.x(), rotation.y(), rotation.z(), rotation.w())
                        .normalize(),
                );
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn cube_falls_and_is_resettable() {
            let mut demo = PhysxDemo::new();
            let start = demo.cube_transform();

            for _ in 0..60 {
                demo.step(1.0 / 60.0);
            }

            let fallen = demo.cube_transform();
            assert!(fallen.translation.is_finite());
            assert!(fallen.translation.y < start.translation.y - 1.0);
            assert!(fallen.translation.y > 0.45);

            demo.reset();
            let reset = demo.cube_transform();
            assert!((reset.translation - start.translation).length() < 1.0e-5);
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod platform {
    use super::*;
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen(module = "/src/physx_bridge.js")]
    extern "C" {
        fn physx_begin_load();
        fn physx_reset();
        fn physx_step(dt: f32);
        fn physx_status_code() -> u32;
        fn physx_cube_x() -> f32;
        fn physx_cube_y() -> f32;
        fn physx_cube_z() -> f32;
        fn physx_cube_qx() -> f32;
        fn physx_cube_qy() -> f32;
        fn physx_cube_qz() -> f32;
        fn physx_cube_qw() -> f32;
    }

    pub struct PhysxDemo {
        cube_transform: Transform,
    }

    impl PhysxDemo {
        pub fn new() -> Self {
            physx_begin_load();
            Self {
                cube_transform: Transform::from_translation(STARTING_CUBE_POSITION),
            }
        }

        pub fn reset(&mut self) {
            self.cube_transform = Transform::from_translation(STARTING_CUBE_POSITION);
            physx_reset();
            self.refresh_cube_transform();
        }

        pub fn step(&mut self, dt: f32) {
            physx_step(dt);
            self.refresh_cube_transform();
        }

        pub fn cube_transform(&self) -> Transform {
            self.cube_transform
        }

        pub fn status(&self) -> &'static str {
            match physx_status_code() {
                1 => "READY",
                2 => "ERROR",
                _ => "LOADING",
            }
        }

        fn refresh_cube_transform(&mut self) {
            if physx_status_code() != 1 {
                return;
            }

            self.cube_transform =
                Transform::from_xyz(physx_cube_x(), physx_cube_y(), physx_cube_z()).with_rotation(
                    Quat::from_xyzw(
                        physx_cube_qx(),
                        physx_cube_qy(),
                        physx_cube_qz(),
                        physx_cube_qw(),
                    )
                    .normalize(),
                );
        }
    }
}

pub use platform::PhysxDemo;

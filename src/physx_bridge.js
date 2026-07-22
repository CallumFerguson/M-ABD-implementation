let PhysX = null;
let allocator = null;
let errorCallback = null;
let foundation = null;
let physics = null;
let scale = null;
let dispatcher = null;
let world = null;
let statusCode = 0;
let loadPromise = null;
let resetWhenReady = false;
let cachedPose = [0, 3, 0, 0, 0, 0, 1];

export function physx_begin_load() {
  if (loadPromise) {
    return;
  }

  const moduleUrl = new URL("physx-js-webidl.mjs", document.baseURI);
  loadPromise = import(moduleUrl.href)
    .then(({ default: loadPhysX }) => loadPhysX())
    .then((module) => {
      PhysX = module;
      allocator = new PhysX.PxDefaultAllocator();
      errorCallback = new PhysX.PxDefaultErrorCallback();
      foundation = PhysX.CreateFoundation(
        PhysX.PHYSICS_VERSION,
        allocator,
        errorCallback,
      );
      scale = new PhysX.PxTolerancesScale();
      physics = PhysX.CreatePhysics(PhysX.PHYSICS_VERSION, foundation, scale);
      dispatcher = PhysX.DefaultCpuDispatcherCreate(0);
      createWorld();
      statusCode = 1;

      if (resetWhenReady) {
        resetWhenReady = false;
        createWorld();
      }
    })
    .catch((error) => {
      statusCode = 2;
      console.error("Unable to initialize PhysX:", error);
    });
}

export function physx_reset() {
  if (statusCode === 1) {
    createWorld();
  } else {
    resetWhenReady = true;
  }
}

export function physx_step(dt) {
  if (statusCode !== 1 || !world) {
    return;
  }

  world.scene.simulate(dt);
  if (!world.scene.fetchResults(true)) {
    statusCode = 2;
    console.error("PhysX failed to fetch simulation results");
    return;
  }
  cacheCubePose();
}

export function physx_status_code() {
  return statusCode;
}

export function physx_cube_x() { return cachedPose[0]; }
export function physx_cube_y() { return cachedPose[1]; }
export function physx_cube_z() { return cachedPose[2]; }
export function physx_cube_qx() { return cachedPose[3]; }
export function physx_cube_qy() { return cachedPose[4]; }
export function physx_cube_qz() { return cachedPose[5]; }
export function physx_cube_qw() { return cachedPose[6]; }

function createWorld() {
  releaseWorld();

  const sceneDesc = new PhysX.PxSceneDesc(scale);
  const gravity = new PhysX.PxVec3(0, -9.81, 0);
  sceneDesc.set_gravity(gravity);
  sceneDesc.set_cpuDispatcher(dispatcher);
  sceneDesc.set_filterShader(PhysX.DefaultFilterShader());
  const scene = physics.createScene(sceneDesc);
  const material = physics.createMaterial(0.55, 0.45, 0.05);
  const shapeFlags = new PhysX.PxShapeFlags(
    PhysX.PxShapeFlagEnum.eSCENE_QUERY_SHAPE |
      PhysX.PxShapeFlagEnum.eSIMULATION_SHAPE |
      PhysX.PxShapeFlagEnum.eVISUALIZATION,
  );
  const filterData = new PhysX.PxFilterData(1, 1, 0, 0);

  const groundGeometry = new PhysX.PxBoxGeometry(3, 0.5, 3);
  const groundPosition = new PhysX.PxVec3(0, -0.5, 0);
  const groundPose = new PhysX.PxTransform(PhysX.PxIDENTITYEnum.PxIdentity);
  groundPose.set_p(groundPosition);
  const groundShape = physics.createShape(groundGeometry, material, true, shapeFlags);
  groundShape.setSimulationFilterData(filterData);
  const ground = physics.createRigidStatic(groundPose);
  ground.attachShape(groundShape);
  groundShape.release();
  scene.addActor(ground);

  const cubeGeometry = new PhysX.PxBoxGeometry(0.5, 0.5, 0.5);
  const cubePosition = new PhysX.PxVec3(0, 3, 0);
  const cubePose = new PhysX.PxTransform(PhysX.PxIDENTITYEnum.PxIdentity);
  cubePose.set_p(cubePosition);
  const cubeShape = physics.createShape(cubeGeometry, material, true, shapeFlags);
  cubeShape.setSimulationFilterData(filterData);
  const cube = physics.createRigidDynamic(cubePose);
  cube.attachShape(cubeShape);
  cubeShape.release();
  scene.addActor(cube);

  PhysX.destroy(cubePose);
  PhysX.destroy(cubePosition);
  PhysX.destroy(cubeGeometry);
  PhysX.destroy(groundPose);
  PhysX.destroy(groundPosition);
  PhysX.destroy(groundGeometry);
  PhysX.destroy(filterData);
  PhysX.destroy(shapeFlags);
  PhysX.destroy(gravity);
  PhysX.destroy(sceneDesc);

  world = { scene, material, ground, cube };
  cachedPose = [0, 3, 0, 0, 0, 0, 1];
  cacheCubePose();
}

function releaseWorld() {
  if (!world) {
    return;
  }

  world.cube.release();
  world.ground.release();
  world.scene.release();
  world.material.release();
  world = null;
}

function cacheCubePose() {
  const pose = world.cube.getGlobalPose();
  const position = pose.get_p();
  const rotation = pose.get_q();
  cachedPose = [
    position.get_x(),
    position.get_y(),
    position.get_z(),
    rotation.get_x(),
    rotation.get_y(),
    rotation.get_z(),
    rotation.get_w(),
  ];
}

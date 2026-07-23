const GRID_SPACING = 0.55;
const SUPPORTED_GRID_SIZES = new Set([10, 25, 50, 100]);
const ROD_LENGTH_FACTOR = 0.78;
const HUB_RADIUS = 0.075;
const ROD_RADIUS = 0.055 * 0.5;
const BALL_RADIUS = 0.34;
const CYLINDER_RADIUS = 0.70;
const CYLINDER_LENGTH_PADDING = 0.85;
const CYLINDER_CENTER_OFFSET = 0.175;
const HUB_MASS = 0.18;
const ROD_MASS = 0.24;
const BALL_MASS = 1.2;
const BODY_DAMPING = -Math.log(0.997) * 30.0;

const SCENE_GRID = 1;
const SCENE_CYLINDER = 2;
const SCENE_BALLS = 3;
const IDENTITY_ROTATION = [0, 0, 0, 1];
const X_TO_NEGATIVE_Z = [0, Math.SQRT1_2, 0, Math.SQRT1_2];
const X_TO_POSITIVE_Z = [0, -Math.SQRT1_2, 0, Math.SQRT1_2];
const EMPTY_TRANSFORMS = new Float32Array(0);

let PhysX = null;
let allocator = null;
let errorCallback = null;
let foundation = null;
let physics = null;
let scale = null;
let dispatcher = null;
let world = null;
let transforms = EMPTY_TRANSFORMS;
let statusCode = 0;
let loadPromise = null;
let pendingConfig = null;

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

      if (!PhysX.InitExtensions(physics)) {
        throw new Error("PhysX extensions failed to initialize");
      }

      statusCode = 1;
      if (pendingConfig !== null) {
        const config = pendingConfig;
        pendingConfig = null;
        createWorld(config.scene, config.gridSize);
      }
    })
    .catch((error) => fail("Unable to initialize PhysX", error));
}

export function physx_reset(sceneId, gridSize) {
  const normalizedScene = normalizeScene(sceneId);
  const normalizedGridSize = normalizeGridSize(gridSize);
  if (statusCode === 1) {
    try {
      createWorld(normalizedScene, normalizedGridSize);
    } catch (error) {
      fail("Unable to create PhysX scene", error);
    }
  } else if (statusCode === 0) {
    pendingConfig = {
      scene: normalizedScene,
      gridSize: normalizedGridSize,
    };
  }
}

export function physx_clear() {
  pendingConfig = null;
  releaseWorld();
}

// This function intentionally performs only simulation work. Transform
// extraction is kept in physx_refresh_transforms() so SIM STEP measures the
// PhysX simulate/fetch pair rather than rendering synchronization.
export function physx_step(dt) {
  if (statusCode !== 1 || !world || !(dt > 0)) {
    return;
  }

  try {
    world.scene.simulate(dt);
    if (!world.scene.fetchResults(true)) {
      throw new Error("fetchResults returned false");
    }
    world.transformsDirty = true;
  } catch (error) {
    fail("PhysX simulation failed", error);
  }
}

export function physx_refresh_transforms() {
  if (statusCode !== 1 || !world || !world.transformsDirty) {
    return;
  }

  for (let index = 0; index < world.renderActors.length; index += 1) {
    // WebIDL marks getGlobalPose() as a value return backed by a reusable
    // temporary. Read it immediately and do not destroy the returned wrappers.
    const pose = world.renderActors[index].getGlobalPose();
    const position = pose.get_p();
    const rotation = pose.get_q();
    const offset = index * 7;

    transforms[offset] = position.get_x();
    transforms[offset + 1] = position.get_y();
    transforms[offset + 2] = position.get_z();
    transforms[offset + 3] = rotation.get_x();
    transforms[offset + 4] = rotation.get_y();
    transforms[offset + 5] = rotation.get_z();
    transforms[offset + 6] = rotation.get_w();
  }

  world.transformsDirty = false;
}

export function physx_transforms() {
  return transforms;
}

export function physx_status_code() {
  return statusCode;
}

export function physx_body_count() {
  return world ? world.renderActors.length : 0;
}

function normalizeScene(sceneId) {
  const scene = Math.trunc(Number(sceneId));
  if (scene < SCENE_GRID || scene > SCENE_BALLS) {
    throw new RangeError(`Unsupported PhysX scene: ${sceneId}`);
  }
  return scene;
}

function normalizeGridSize(gridSize) {
  const size = Math.trunc(Number(gridSize));
  if (!SUPPORTED_GRID_SIZES.has(size)) {
    throw new RangeError(`Unsupported PhysX grid size: ${gridSize}`);
  }
  return size;
}

function netBodyCount(gridSize) {
  return 3 * gridSize * gridSize - 2 * gridSize;
}

function netJointCount(gridSize) {
  return 4 * gridSize * (gridSize - 1);
}

function createWorld(sceneId, gridSize) {
  releaseWorld();

  const bodyCount = netBodyCount(gridSize);

  const next = {
    scene: null,
    material: null,
    aggregate: null,
    joints: [],
    ownedActors: [],
    renderActors: [],
    transformsDirty: true,
    filterData: null,
  };
  let shapeFlags = null;
  let filterData = null;

  try {
    const sceneDesc = new PhysX.PxSceneDesc(scale);
    const gravity = new PhysX.PxVec3(0, -9.81, 0);
    sceneDesc.set_gravity(gravity);
    sceneDesc.set_cpuDispatcher(dispatcher);
    sceneDesc.set_filterShader(PhysX.DefaultFilterShader());
    sceneDesc.set_solverType(PhysX.PxSolverTypeEnum.eTGS);
    next.scene = physics.createScene(sceneDesc);
    PhysX.destroy(gravity);
    PhysX.destroy(sceneDesc);

    if (!next.scene || next.scene.ptr === 0) {
      throw new Error("createScene returned null");
    }

    next.material = physics.createMaterial(0.25, 0.20, 0.03);
    next.aggregate = physics.createAggregate(
      bodyCount,
      bodyCount,
      false,
    );
    shapeFlags = new PhysX.PxShapeFlags(
      PhysX.PxShapeFlagEnum.eSCENE_QUERY_SHAPE |
        PhysX.PxShapeFlagEnum.eSIMULATION_SHAPE,
    );
    // PxDefaultSimulationFilterShader expects compatible group/mask words.
    // The aggregate still suppresses net-vs-net pairs, while this enables the
    // net to contact the cylinder and falling balls.
    filterData = new PhysX.PxFilterData(1, 1, 0, 0);
    next.filterData = filterData;

    buildJointNet(next, shapeFlags, sceneId, gridSize);
    if (!next.scene.addAggregate(next.aggregate)) {
      throw new Error("Unable to add joint-net aggregate to the scene");
    }

    if (sceneId === SCENE_CYLINDER) {
      buildCylinder(next, shapeFlags, gridSize);
    } else if (sceneId === SCENE_BALLS) {
      buildBalls(next, shapeFlags);
    }

    world = next;
    transforms = new Float32Array(next.renderActors.length * 7);
    physx_refresh_transforms();
  } catch (error) {
    disposeWorld(next);
    transforms = EMPTY_TRANSFORMS;
    throw error;
  } finally {
    if (shapeFlags) {
      PhysX.destroy(shapeFlags);
    }
    if (filterData) {
      PhysX.destroy(filterData);
      next.filterData = null;
    }
  }
}

function buildJointNet(next, shapeFlags, sceneId, gridSize) {
  const hubs = Array.from({ length: gridSize }, () =>
    new Array(gridSize),
  );
  const positions = Array.from({ length: gridSize }, () =>
    new Array(gridSize),
  );
  const gridHeight = sceneId === SCENE_BALLS ? 1.70 : 3.0;
  const hubGeometry = new PhysX.PxSphereGeometry(HUB_RADIUS);
  const rodGeometry = new PhysX.PxBoxGeometry(
    GRID_SPACING * ROD_LENGTH_FACTOR * 0.5,
    ROD_RADIUS,
    ROD_RADIUS,
  );

  try {
    // Hubs are first, matching NetSimulation's body ordering.
    for (let row = 0; row < gridSize; row += 1) {
      for (let column = 0; column < gridSize; column += 1) {
        const position = [
          (column - (gridSize - 1) * 0.5) * GRID_SPACING,
          gridHeight,
          sceneId === SCENE_BALLS
            ? (row - (gridSize - 1) * 0.5) * GRID_SPACING
            : -row * GRID_SPACING,
        ];
        const fixed =
          sceneId === SCENE_BALLS
            ? (row === 0 || row === gridSize - 1) &&
              (column === 0 || column === gridSize - 1)
            : row === 0;
        const hub = createActor(
          next,
          shapeFlags,
          hubGeometry,
          position,
          IDENTITY_ROTATION,
          HUB_MASS,
          fixed,
          true,
        );

        positions[row][column] = position;
        hubs[row][column] = hub;
        next.renderActors.push(hub);
      }
    }

    // Horizontal rods follow the hubs.
    for (let row = 0; row < gridSize; row += 1) {
      for (let column = 0; column < gridSize - 1; column += 1) {
        addRod(
          next,
          shapeFlags,
          rodGeometry,
          hubs[row][column],
          hubs[row][column + 1],
          positions[row][column],
          positions[row][column + 1],
          IDENTITY_ROTATION,
        );
      }
    }

    // Vertical rods follow the horizontal rods.
    const verticalRotation =
      sceneId === SCENE_BALLS ? X_TO_POSITIVE_Z : X_TO_NEGATIVE_Z;
    for (let row = 0; row < gridSize - 1; row += 1) {
      for (let column = 0; column < gridSize; column += 1) {
        addRod(
          next,
          shapeFlags,
          rodGeometry,
          hubs[row][column],
          hubs[row + 1][column],
          positions[row][column],
          positions[row + 1][column],
          verticalRotation,
        );
      }
    }
  } finally {
    PhysX.destroy(rodGeometry);
    PhysX.destroy(hubGeometry);
  }

  if (
    next.renderActors.length !== netBodyCount(gridSize) ||
    next.joints.length !== netJointCount(gridSize)
  ) {
    throw new Error("Joint-net construction produced unexpected counts");
  }
}

function addRod(
  next,
  shapeFlags,
  geometry,
  startHub,
  endHub,
  start,
  end,
  rotation,
) {
  const midpoint = [
    (start[0] + end[0]) * 0.5,
    (start[1] + end[1]) * 0.5,
    (start[2] + end[2]) * 0.5,
  ];
  const rod = createActor(
    next,
    shapeFlags,
    geometry,
    midpoint,
    rotation,
    ROD_MASS,
    false,
    true,
  );

  next.renderActors.push(rod);
  createSphericalJoint(next, rod, startHub, -GRID_SPACING * 0.5);
  createSphericalJoint(next, rod, endHub, GRID_SPACING * 0.5);
}

function createSphericalJoint(next, rod, hub, rodOffset) {
  const rodFrame = makePose([rodOffset, 0, 0], IDENTITY_ROTATION);
  const hubFrame = makePose([0, 0, 0], IDENTITY_ROTATION);
  const joint = PhysX.SphericalJointCreate(
    physics,
    rod,
    rodFrame,
    hub,
    hubFrame,
  );
  PhysX.destroy(hubFrame);
  PhysX.destroy(rodFrame);

  if (!joint || joint.ptr === 0) {
    throw new Error("SphericalJointCreate returned null");
  }
  next.joints.push(joint);
}

function buildCylinder(next, shapeFlags, gridSize) {
  const gridSpan = (gridSize - 1) * GRID_SPACING;
  const cylinderLength = gridSpan + CYLINDER_LENGTH_PADDING;
  const cylinderZ = -gridSpan * 0.5 + CYLINDER_CENTER_OFFSET;
  // PxCapsuleGeometry is X-aligned; subtracting the radius makes its total
  // end-to-end length match the grid-width cylinder visual.
  const geometry = new PhysX.PxCapsuleGeometry(
    CYLINDER_RADIUS,
    cylinderLength * 0.5 - CYLINDER_RADIUS,
  );
  try {
    const cylinder = createActor(
      next,
      shapeFlags,
      geometry,
      [0, 1.75, cylinderZ],
      IDENTITY_ROTATION,
      0,
      true,
      false,
    );
    if (!next.scene.addActor(cylinder)) {
      throw new Error("Unable to add cylinder actor to the scene");
    }
  } finally {
    PhysX.destroy(geometry);
  }
}

function buildBalls(next, shapeFlags) {
  const geometry = new PhysX.PxSphereGeometry(BALL_RADIUS);
  const ballPositions = [
    [-1.10, 2.80, -0.80],
    [0.85, 3.15, -0.25],
    [-0.20, 3.50, 1.05],
  ];

  try {
    // Balls follow all net bodies, matching NetSimulation.
    for (const position of ballPositions) {
      const ball = createActor(
        next,
        shapeFlags,
        geometry,
        position,
        IDENTITY_ROTATION,
        BALL_MASS,
        false,
        false,
      );
      if (!next.scene.addActor(ball)) {
        throw new Error("Unable to add ball actor to the scene");
      }
      next.renderActors.push(ball);
    }
  } finally {
    PhysX.destroy(geometry);
  }
}

function createActor(
  next,
  shapeFlags,
  geometry,
  position,
  rotation,
  mass,
  fixed,
  addToAggregate,
) {
  const pose = makePose(position, rotation);
  const shape = physics.createShape(geometry, next.material, true, shapeFlags);
  const actor = fixed
    ? physics.createRigidStatic(pose)
    : physics.createRigidDynamic(pose);
  PhysX.destroy(pose);

  if (!shape || shape.ptr === 0 || !actor || actor.ptr === 0) {
    if (shape && shape.ptr !== 0) {
      shape.release();
    }
    if (actor && actor.ptr !== 0) {
      actor.release();
    }
    throw new Error("Unable to create PhysX actor or shape");
  }

  if (!actor.attachShape(shape)) {
    shape.release();
    actor.release();
    throw new Error("Unable to attach PhysX shape");
  }
  shape.setSimulationFilterData(next.filterData);
  shape.release();

  if (!fixed) {
    // Static extension functions are emitted on the prototype by WebIDL.
    if (!PhysX.PxRigidBodyExt.prototype.setMassAndUpdateInertia(actor, mass)) {
      actor.release();
      throw new Error("Unable to set PhysX mass and inertia");
    }
    actor.setLinearDamping(BODY_DAMPING);
    actor.setAngularDamping(BODY_DAMPING);
    actor.setSolverIterationCounts(16, 4);
  }

  next.ownedActors.push(actor);
  if (addToAggregate && !next.aggregate.addActor(actor)) {
    next.ownedActors.pop();
    actor.release();
    throw new Error("Unable to add actor to joint-net aggregate");
  }
  return actor;
}

function makePose(position, rotation) {
  const vector = new PhysX.PxVec3(position[0], position[1], position[2]);
  const quaternion = new PhysX.PxQuat(
    rotation[0],
    rotation[1],
    rotation[2],
    rotation[3],
  );
  const pose = new PhysX.PxTransform(vector, quaternion);
  PhysX.destroy(quaternion);
  PhysX.destroy(vector);
  return pose;
}

function releaseWorld() {
  if (!world) {
    transforms = EMPTY_TRANSFORMS;
    return;
  }

  const oldWorld = world;
  world = null;
  transforms = EMPTY_TRANSFORMS;
  disposeWorld(oldWorld);
}

function disposeWorld(target) {
  for (let index = target.joints.length - 1; index >= 0; index -= 1) {
    target.joints[index].release();
  }
  for (let index = target.ownedActors.length - 1; index >= 0; index -= 1) {
    target.ownedActors[index].release();
  }
  if (target.aggregate) {
    target.aggregate.release();
  }
  if (target.scene) {
    target.scene.release();
  }
  if (target.material) {
    target.material.release();
  }
}

function fail(message, error) {
  statusCode = 2;
  console.error(`${message}:`, error);
}

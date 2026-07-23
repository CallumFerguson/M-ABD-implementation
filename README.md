# M-ABD and PhysX comparison

A focused proof-of-concept implementation of the ball-joint nets from
*M-ABD: Scalable, Efficient, and Robust Multi-Affine-Body Dynamics*.

The application has three scenes that can be changed at runtime. Each scene can
run either the project's M-ABD-inspired solver or an independently constructed
PhysX version. Changing the scene, changing the backend, or resetting rebuilds
the selected simulation from its initial state:

- `1`: the original edge-pinned 10x10 joint grid
- `2`: the edge-pinned joint grid draped over a static cylinder
- `3`: a horizontal, four-corner-pinned joint grid catching three falling balls
- `B`: switch between the project solver and PhysX
- `R`: reset the current scene
- `10x10`, `25x25`, `50x50`, and `100x100` buttons: rebuild the current
  scene/backend with that grid size
- `30`, `60`, `120`, `200`, and `500` Hz buttons: change the fixed simulation
  rate without resetting the scene

The default 10x10 net contains the paper's inferred 280-body topology:

- 100 affine hub bodies arranged in a 10x10 grid
- 180 affine rod bodies joining horizontal and vertical neighbors
- 360 ball joints, or 1,080 scalar positional constraints
- either one fixed edge (10 hubs at the default size) or four fixed corner hubs

For an `N x N` grid, the same topology scales to `3N² - 2N` bodies and
`4N(N - 1)` ball joints. Grid-size changes preserve the existing per-cell
spacing, resize the cylinder scene to span the wider net, reframe the camera,
and reset the simulation.

Each affine body is represented by the four control points from Section 4.1.
Each fixed step performs an implicit prediction and a compact co-rotated
local/global solve, using a matrix-free dual KKT solve for all linear ball-joint
constraints. Primitive sphere/capsule contacts provide two-way interaction with
the cylinder and falling balls. The default timestep is the Figure 12 value of
`1/30 s`; the on-screen controls can change it up to `1/500 s` at runtime. A
compact overlay reports the active scene, smoothed FPS and frame time, measured
simulation-step duration, fixed rate, grid dimensions, and scene size.

The PhysX versions manually recreate all three setups with rigid sphere and box
actors connected by spherical joints, plus a static capsule for the cylinder
scene and rigid spheres for the falling-ball scene. There is no live state
transfer or shared physics-scene format between the backends. Bevy owns the
window, rendering, input, scene switching, and overlay; the active backend owns
the simulation state. Native builds use the `physx` Rust wrapper (PhysX 5.1.3),
while browser builds use the `physx-js-webidl` WebAssembly package (PhysX
5.6.1).

## Run natively

The first native build compiles the PhysX C++ SDK and therefore requires a C++
toolchain. On Windows, use the MSVC Rust toolchain with Visual Studio Build
Tools and the **Desktop development with C++** workload.

```sh
cargo run
```

## Run in a browser

Install the one-time prerequisites if needed:

```sh
rustup target add wasm32-unknown-unknown
cargo install --locked trunk
npm install
```

Start the development server:

```sh
npm run serve -- --open
```

The site is available at <http://127.0.0.1:8080> if it does not open
automatically. Create a deployable build with:

```sh
npm run build
```

## MVP scope

This demo includes co-rotated affine bodies, fixed-step implicit prediction,
linear ball joints, a dual constraint solve, and deliberately narrow analytic
contacts for spheres, rod capsules, and one static cylinder. It intentionally
omits friction, restitution, continuous collision detection, self-collision,
arbitrary mesh collision, other joint types, interactive manipulation, GPU
compute, and the paper's million-body solver optimizations. The PhysX versions
are deliberately narrow equivalents of these three demos, not a general
scene-conversion layer.

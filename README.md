# M-ABD and PhysX demos

A focused proof-of-concept implementation of the ball-joint nets from
*M-ABD: Scalable, Efficient, and Robust Multi-Affine-Body Dynamics*.

The application has four scenes that can be changed at runtime. Changing or
resetting a scene rebuilds that scene's simulation:

- `1`: the original edge-pinned 10x10 joint grid
- `2`: the edge-pinned joint grid draped over a static cylinder
- `3`: a horizontal, four-corner-pinned joint grid catching three falling balls
- `4`: a PhysX smoke test with one dynamic cube falling onto one fixed cube
- `R`: reset the current scene
- `30`, `60`, `120`, `200`, and `500` Hz buttons: change the fixed simulation
  rate without resetting the scene

Each net contains the paper's inferred 280-body topology:

- 100 affine hub bodies arranged in a 10x10 grid
- 180 affine rod bodies joining horizontal and vertical neighbors
- 360 ball joints, or 1,080 scalar positional constraints
- either 10 fixed hubs along one edge or four fixed corner hubs

Each affine body is represented by the four control points from Section 4.1.
Each fixed step performs an implicit prediction and a compact co-rotated
local/global solve, using a matrix-free dual KKT solve for all linear ball-joint
constraints. Primitive sphere/capsule contacts provide two-way interaction with
the cylinder and falling balls. The default timestep is the Figure 12 value of
`1/30 s`; the on-screen controls can change it up to `1/500 s` at runtime. A
compact overlay reports the active scene, smoothed FPS and frame time, measured
simulation-step duration, fixed rate, and scene size.

Scene 4 is deliberately independent of the M-ABD implementation. Bevy owns the
window, rendering, input, scene switching, and overlay, while PhysX owns the two
rigid bodies and supplies the falling cube's pose. Native builds use the
`physx` Rust wrapper (PhysX 5.1.3); browser builds use the official
`physx-js-webidl` WebAssembly package (PhysX 5.6.1).

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
arbitrary mesh collision, other joint types, controls, GPU compute, and the
paper's million-body solver optimizations. The PhysX scene is only a two-body
integration check; it does not yet recreate any of the paper scenes in PhysX.

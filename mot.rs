//! A 2D+ mot configuration, loaded directly from oven.

extern crate atomecs;
extern crate nalgebra;
use atomecs::atom::{Atom, Force, Mass};
use atomecs::atom::{AtomicTransition, Position, Velocity};
use atomecs::atom_sources::emit::AtomNumberToEmit;
use atomecs::atom_sources::mass::{MassDistribution, MassRatio};
use atomecs::atom_sources::oven::{OvenAperture, OvenBuilder};
use atomecs::atom_sources::VelocityCap;
use atomecs::destructor::ToBeDestroyed;
use atomecs::ecs;
use atomecs::initiate::NewlyCreated;
use atomecs::integrator::Timestep;
use atomecs::laser::gaussian::GaussianBeam;
use atomecs::laser_cooling::force::{EmissionForceConfiguration, EmissionForceOption};
use atomecs::laser_cooling::photons_scattered::ScatteringFluctuationsOption;
use atomecs::laser_cooling::CoolingLight;
use atomecs::magnetic::grid::PrecalculatedMagneticFieldGrid;
use atomecs::magnetic::quadrupole::{QuadrupoleField2D, QuadrupoleField3D};
use atomecs::magnetic::uniform::UniformMagneticField;
use atomecs::output::file;
use atomecs::output::file::Text;
use atomecs::shapes::{Cuboid, Cylinder};
use atomecs::sim_region::{SimulationVolume, VolumeType};
use nalgebra::{Unit, Vector3};
use specs::prelude::*;
use std::fs::read_to_string;
use std::time::Instant;
use std::fs::File;
use std::io::{BufReader,BufWriter};
use std::path::Path;
use std::io::Write;

//alex changes
use rayon::ThreadPoolBuilder;
use std::env;
//////////

extern crate serde;
use serde::Deserialize;
  
/// Parameters describing this simulation
#[derive(Deserialize)]
pub struct SimulationParameters {
    /// Radius of the push beam, units of mm
    pub push_beam_radius: f64,
    /// Power of the push beam, units of mW
    pub push_beam_power: f64,
    /// Detuning of the push beam, units of MHz
    pub push_beam_detuning: f64,
    /// Detuning of the cooling beams, units of MHz, eg: -45.
    pub cooling_beam_detuning: f64,
    /// Gradient of the quadrupole field, Gauss/cm. eg: 65.
    pub quadrupole_gradient: f64,
    /// 1/e radius of the cooling_beam, units of mm
    pub cooling_beam_radius: f64,
    /// Offset of the push beam from the quadrupole node, units of mm.
    pub push_beam_offset: f64,

    /// strength of a bias field in the vertical direction, G
    pub vertical_bias_field: f64,

    /// The number of atoms to simulate. 4e6
    pub atom_number: i32,

    /// Velocity cap of atoms leaving the oven. 230m/s
    pub oven_velocity_cap: f64,

    /// x position of the oven, in mm.
    pub oven_position_x_mm: f64,

    /// y position of the oven, in mm.
    pub oven_position_y_mm: f64,

    /// z position of the oven, in mm.
    pub oven_position_z_mm: f64,

    /// Generates atoms at a well-defined position with deterministic velocity, instead of using the oven.
    /// This switch is used to enable phase plots of (z,v_z)
    pub phase_plot: bool,

    /// If true, use a 3D quadrupole of the form B'(x, y, -2z).
    /// If false, use a 2D quadrupole of the form B'(x, -y)
    pub use_3d_quadrupole: bool,

    /// Radius of microchannels in oven nozzle, units of mm. Current Ox oven uses 0.2mm.
    pub microchannel_radius: f64,

    /// Length of microchannels in oven nozzle, units of mm. Current Ox oven uses 4mm.
    pub microchannel_length: f64,

    /// Radius of the differential pumping section, in mm.
    pub diff_pump_radius_mm: f64,

    /// Offset of the differential pumping aperture, in mm.
    pub diff_pump_offset_mm: f64,

    /// Intersection of the cooling beams, offset along vertical, in mm
    pub cooling_intersection_offset_mm: f64,

    /// Power of the cooling beam in mW
    pub cooling_beam_power_mw: f64,

    pub zeeman_slower_radius_mm: f64,

    pub zeeman_slower_detuning_mhz: f64,

    pub zeeman_slower_power_mw: f64,

    pub use_zeeman_slower: bool,

    /// Whether to use a pre-computed grid for the magnetic field
    pub use_field_grid: bool,
}

fn main() {

    //alex changes
    ThreadPoolBuilder::new().num_threads(8).build_global().unwrap();

    env::set_var("OPENBLAS_NUM_THREADS", "8");
    env::set_var("MKL_NUM_THREADS", "8");
    env::set_var("OMP_NUM_THREADS", "8");
    ////////////

    let now = Instant::now();

    let json_str = read_to_string("input.json").expect("Could not open file");
    println!("Loaded json string: {}", json_str);
    let parameters: SimulationParameters = serde_json::from_str(&json_str).unwrap();

    // Create the simulation world and builder for the ECS dispatcher.
    let mut world = World::new();
    ecs::register_components(&mut world);
    ecs::register_resources(&mut world);
    let mut builder = ecs::create_simulation_dispatcher_builder();

    // Configure simulation output.
    builder = builder.with(
        file::new::<Position, Text>("pos.txt".to_string(), 64),
        "",
        &[],
    );
    builder = builder.with(
        file::new::<Velocity, Text>("vel.txt".to_string(), 64),
        "",
        &[],
    );

    builder = builder.with(
        InitialAtomVelocitySystem::new("initial_v.txt".to_string()),
        "initial_atom_velocity_system",
        &[]
    );

    let mut dispatcher = builder.build();
    dispatcher.setup(&mut world);

    // Create magnetic field.
    if !parameters.use_field_grid {
        if parameters.use_3d_quadrupole {
            world
                .create_entity()
                .with(QuadrupoleField3D::gauss_per_cm(
                    parameters.quadrupole_gradient,
                    Vector3::z(),
                ))
                .with(Position::new())
                .build();
        } else {
            let out_direction = Unit::new_normalize(Vector3::new(1.0, 1.0, 0.0).normalize());
            world
                .create_entity()
                .with(QuadrupoleField2D::gauss_per_cm(
                    parameters.quadrupole_gradient,
                    Vector3::z_axis(),
                    out_direction,
                ))
                .with(Position::new())
                .build();
        }
    } else {
        let f = File::open("field.json").expect("Could not open file.");
        let reader = BufReader::new(f);
        let grid: PrecalculatedMagneticFieldGrid = serde_json::from_reader(reader)
            .expect("Could not load magnetic field grid from json file.");
        world.create_entity().with(grid).build();
    }

    world
        .create_entity()
        .with(UniformMagneticField::gauss(Vector3::new(
            parameters.vertical_bias_field,
            0.0,
            0.0,
        )))
        .build();

    // Create push beam
    world
        .create_entity()
        .with(GaussianBeam {
            intersection: Vector3::new(parameters.push_beam_offset * 1.0e-3, 0.0, 0.0),
            e_radius: parameters.push_beam_radius * 1.0e-3,
            power: parameters.push_beam_power * 1.0e-3,
            direction: Vector3::z(),
            rayleigh_range: f64::INFINITY,
            ellipticity: 0.0,
        })
        .with(CoolingLight::for_species(
            AtomicTransition::strontium(),
            parameters.push_beam_detuning,
            -1,
        ))
        .build();

    // Create cooling lasers. Note that one polarisation swaps depending on whether we have a 2D or 3D config.
    let detuning = parameters.cooling_beam_detuning;
    let power = parameters.cooling_beam_power_mw * 1.0e-3;
    let radius = parameters.cooling_beam_radius * 1.0e-3;
    let polarisation11 = if parameters.use_3d_quadrupole { 1 } else { 1 };
    let polarisation10 = if parameters.use_3d_quadrupole { 1 } else { -1 };
    world
        .create_entity()
        .with(GaussianBeam {
            intersection: Vector3::new(
                parameters.cooling_intersection_offset_mm * 1.0e-3,
                0.0,
                0.0,
            ),
            e_radius: radius,
            power: power,
            direction: Vector3::new(1.0, 1.0, 0.0).normalize(),
            rayleigh_range: f64::INFINITY,
            ellipticity: 0.0,
        })
        .with(CoolingLight::for_species(
            AtomicTransition::strontium(),
            detuning,
            polarisation11,
        ))
        .build();
    world
        .create_entity()
        .with(GaussianBeam {
            intersection: Vector3::new(
                parameters.cooling_intersection_offset_mm * 1.0e-3,
                0.0,
                0.0,
            ),
            e_radius: radius,
            power: power,
            direction: Vector3::new(1.0, -1.0, 0.0).normalize(),
            rayleigh_range: f64::INFINITY,
            ellipticity: 0.0,
        })
        .with(CoolingLight::for_species(
            AtomicTransition::strontium(),
            detuning,
            polarisation10,
        ))
        .build();
    world
        .create_entity()
        .with(GaussianBeam {
            intersection: Vector3::new(
                parameters.cooling_intersection_offset_mm * 1.0e-3,
                0.0,
                0.0,
            ),
            e_radius: radius,
            power: power,
            direction: Vector3::new(-1.0, 1.0, 0.0).normalize(),
            rayleigh_range: f64::INFINITY,
            ellipticity: 0.0,
        })
        .with(CoolingLight::for_species(
            AtomicTransition::strontium(),
            detuning,
            polarisation10,
        ))
        .build();
    world
        .create_entity()
        .with(GaussianBeam {
            intersection: Vector3::new(
                parameters.cooling_intersection_offset_mm * 1.0e-3,
                0.0,
                0.0,
            ),
            e_radius: radius,
            power: power,
            direction: Vector3::new(-1.0, -1.0, 0.0).normalize(),
            rayleigh_range: f64::INFINITY,
            ellipticity: 0.0,
        })
        .with(CoolingLight::for_species(
            AtomicTransition::strontium(),
            detuning,
            polarisation11,
        ))
        .build();

    // zeeman slower
    if parameters.use_zeeman_slower {
        world
            .create_entity()
            .with(GaussianBeam {
                intersection: Vector3::new(0.0, 0.0, 0.0),
                e_radius: parameters.zeeman_slower_radius_mm * 1e-3,
                power: parameters.zeeman_slower_power_mw * 1e-3,
                direction: Vector3::new(-1.0, 0.0, 0.0).normalize(),
                rayleigh_range: f64::INFINITY,
                ellipticity: 0.0,
            })
            .with(CoolingLight::for_species(
                AtomicTransition::strontium(),
                parameters.zeeman_slower_detuning_mhz,
                1,
            ))
            .build();
    }

    if parameters.phase_plot {
        // Create atoms
        for i in 0..38 {
            world
                .create_entity()
                .with(Position {
                    pos: Vector3::new(parameters.oven_position_x_mm * 1e-3, 0.0, 0.0),
                })
                .with(Atom)
                .with(Force::new())
                .with(Velocity {
                    vel: Vector3::new(10.0 + (i as f64) * 5.0, 0.0, 0.0),
                })
                .with(NewlyCreated)
                .with(AtomicTransition::strontium())
                .with(Mass { value: 88.0 })
                .build();
        }
    } else {
        // Create an oven.
        // The oven will eject atoms on the first frame and then be deleted.
        let number_to_emit = parameters.atom_number; //1500000;
        world
            .create_entity()
            .with(
                OvenBuilder::new(776.0, Vector3::x())
                    .with_aperture(OvenAperture::Circular {
                        radius: 0.005,
                        thickness: 0.001,
                    })
                    .with_microchannels(
                        parameters.microchannel_length * 1e-3,
                        parameters.microchannel_radius * 1e-3,
                    )
                    .build(),
            )
            .with(Position {
                pos: Vector3::new(
                    parameters.oven_position_x_mm,
                    parameters.oven_position_y_mm,
                    parameters.oven_position_z_mm,
                ) * 1e-3,
            })
            .with(MassDistribution::new(vec![MassRatio {
                mass: 88.0,
                ratio: 1.0,
            }]))
            .with(AtomicTransition::strontium())
            .with(AtomNumberToEmit {
                number: number_to_emit,
            })
            .with(ToBeDestroyed)
            .build();
    }

    // Use a simulation bound so that atoms that escape the capture region are deleted from the simulation.
    world
        .create_entity()
        .with(Position {
            pos: Vector3::new(0.0, 0.0, 0.0),
        })
        .with(Cuboid {
            half_width: Vector3::new(0.1, 0.03, 0.03),
        })
        .with(SimulationVolume {
            volume_type: VolumeType::Inclusive,
        })
        .build();

    // A differential pumping aperture through which atoms can travel.
    world
        .create_entity()
        .with(Position {
            pos: Vector3::new(parameters.diff_pump_offset_mm * 1.0e-3, 0.0, 0.0),
        })
        .with(Cylinder {
            radius: parameters.diff_pump_radius_mm * 1e-3,
            length: 0.1,
            direction: Vector3::new(0.0, 0.0, 1.0),
            perp_x: Vector3::new(1.0, 0.0, 0.0),
            perp_y: Vector3::new(0.0, 1.0, 0.0),
        })
        .with(SimulationVolume {
            volume_type: VolumeType::Inclusive,
        })
        .build();

    // The simulation bound also now includes a small pipe to capture the 2D MOT output properly.
    world
        .create_entity()
        .with(Position {
            pos: Vector3::new(parameters.diff_pump_offset_mm * 1.0e-3, 0.0, 4.0e-2 + 0.4), //4cm away, 3cm chamber = 1cm DP tube.
        })
        .with(Cuboid {
            half_width: Vector3::new(0.005, 0.005, 0.4),
        })
        .with(SimulationVolume {
            volume_type: VolumeType::Inclusive,
        })
        .build();

    // Also use a velocity cap so that fast atoms are not even simulated.
    world.insert(VelocityCap {
        value: parameters.oven_velocity_cap,
    });

    // Define timestep
    world.insert(Timestep { delta: 2.0e-6 });

    world.insert(EmissionForceOption::On(EmissionForceConfiguration {
        explicit_threshold: 5,
    }));
    world.insert(ScatteringFluctuationsOption::On);

    // Run the simulation for a number of steps.
    for _i in 0..25000 {
        dispatcher.dispatch(&mut world);
        world.maintain();
    }

    println!("Simulation completed in {} ms.", now.elapsed().as_millis());
}

/// Records the initial velocities of atoms generated during the simulation.
pub struct InitialAtomVelocitySystem {
    pub stream: BufWriter<File>,
}
impl<'a> System<'a> for InitialAtomVelocitySystem {
    type SystemData = (
        ReadStorage<'a, Velocity>,
        ReadStorage<'a, Atom>,
        ReadStorage<'a, NewlyCreated>,
    );
    fn run(&mut self, (vel, atom, created): Self::SystemData) {
        for (vel, _, _) in (&vel, &atom, &created).join() {
                write!(self.stream, "{:?}, {:?}, {:?}\n", vel.vel[0], vel.vel[1], vel.vel[2]).expect("Could not write to file");
        }
    }
}

impl InitialAtomVelocitySystem {
    pub fn new(
        file_name: String
    ) -> InitialAtomVelocitySystem
    {
        let path = Path::new(&file_name);
        let display = path.display();
        let file = match File::create(&path) {
            Err(why) => panic!("couldn't open {}: {}", display, why.to_string()),
            Ok(file) => file,
        };
        let writer = BufWriter::new(file);
        InitialAtomVelocitySystem {
            stream: writer
        }
    }
}

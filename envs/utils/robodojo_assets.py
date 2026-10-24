"""Load explicitly prepared RoboDojo geometry into the existing SAPIEN scene.

No downloading, USD parsing, renderer creation, or simulation stepping occurs.
Metadata stays in its original local frames for private task evaluation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path


def effective_source_physics(kind, physics):
    """Reproduce the original importer without rewriting layout or metadata."""
    if kind in ('Rigid', 'Clutter'):
        mass = min(physics.get('mass', .5), .5)
        return {'mass': mass if mass > 0 else .05,
                'static_friction': physics.get('static_friction', .6),
                'dynamic_friction': physics.get('dynamic_friction', 1.5),
                'restitution': physics.get('restitution', 0)}
    if kind == 'Geometry':
        return {'static_friction': physics.get('static_friction', 0),
                'dynamic_friction': physics.get('dynamic_friction', 0),
                'restitution': physics.get('restitution', .5)}
    return {name: physics[name] for name in ('mass', 'static_friction', 'dynamic_friction', 'restitution')
            if name in physics}


class ConvertedAsset:
    def __init__(self, manifest, articulations, root_offset, effective_scale):
        self.metadata = manifest['metadata']
        self.articulations = list(articulations)
        self.articulation = self.articulations[0]
        self.effective_scale = effective_scale
        self.links_by_name = {link.get_name(): link for art in self.articulations for link in art.get_links()}
        self.joints_by_name = {joint.get_name(): joint for art in self.articulations
                               for joint in art.get_joints() if joint.get_name()}
        self._root_offset = root_offset
        self._joint_locations = {}
        for art in self.articulations:
            offset = 0
            for joint in art.get_active_joints():
                self._joint_locations[joint.get_name()] = (art, offset)
                offset += joint.get_dof()

    def get_pose(self):
        return self.articulation.get_root_pose() * self._root_offset.inv()

    def get_link_pose(self, name):
        return self.links_by_name[name].get_pose()

    def get_joint_position(self, name):
        if name not in self.joints_by_name:
            raise KeyError(name)
        if name not in self._joint_locations:
            return 0.0
        art, index = self._joint_locations[name]
        return float(art.get_qpos()[index])

    def get_joint_limits(self, name):
        joint = self.joints_by_name[name]
        return [float(x) for x in joint.get_limits()[0]] if joint.get_dof() else [0.0, 0.0]


def load_converted_asset(scene, manifest_path, pose, *, fixed_root=False, scale=(1, 1, 1), physics=None):
    """Build authored bodies/joints; ``pose`` specifies the original asset frame.

    Source meshes and joint lengths are in metres. Disconnected source bodies
    remain independent articulations; no connecting joint is invented. Moving
    articulations reject anisotropic layout scaling, which would introduce shear.
    """
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema_version') != 'robodojo_sapien_asset_v1':
        raise ValueError('Unsupported converted asset schema; run explicit offline asset preparation')
    import numpy as np
    import sapien

    scene = getattr(scene, 'scene', scene)
    physics = dict(physics or {})
    effective_physics = effective_source_physics(manifest['source']['kind'], physics)
    meters = manifest['units']['meters_per_unit']
    kilograms = manifest['units'].get('kilograms_per_unit', 1.0)
    scale = np.asarray(scale, dtype=float)
    if scale.shape != (3,) or np.any(scale <= 0):
        raise ValueError('Layout scale must contain three positive values')
    movable = [j for j in manifest['joints'] if j['body0'] is not None and j['type'] != 'fixed']
    if movable and not np.allclose(scale, scale[0], atol=1e-10):
        raise ValueError('Anisotropic scale for a moving articulation requires explicit geometric conversion')
    fixed_root = bool(fixed_root or physics.get('fixed_base', False) or manifest['root']['world_fixed_joints'])
    if manifest['source']['kind'] == 'Geometry' and not fixed_root:
        raise ValueError('Original static triangle collision requires fixed_root=True')

    def local_pose(value):
        return sapien.Pose(np.asarray(value['p']) * scale, value['q'])

    axis_rotation = {'X': [1, 0, 0, 0], 'Y': [math.sqrt(.5), 0, 0, math.sqrt(.5)],
                     'Z': [math.sqrt(.5), 0, -math.sqrt(.5), 0]}
    links = {link['source_path']: link for link in manifest['links']}
    root_path = manifest['root']['link']
    components = manifest['root'].get('components', [root_path])
    disconnected = len(components) > 1
    outgoing = {path: [] for path in links}
    for joint in manifest['joints']:
        if joint['body0'] is not None:
            outgoing[joint['body0']].append(joint)
    source_joints = {j['name']: j for j in manifest['joints']}
    created, collision_counts, initial_positions, initial_velocities = {}, {}, {}, {}
    all_volume = sum(m['collision_volume'] for link in links.values()
                     for m in link['meshes'] if m['collision']) * float(np.prod(scale))
    effective_masses, authored_inertia_links = {}, set()

    def add_link(builder, path, parent=None, incoming=None):
        if path in created:
            raise ValueError(f'USD joint graph has a loop or multiple parents at {path}')
        link = links[path]
        link_builder = builder.create_link_builder(parent)
        link_builder.set_name(link['name'])
        created[path] = link_builder
        if not incoming and manifest['root']['world_fixed_joints']:
            link_builder.set_joint_name(manifest['root']['world_fixed_joints'][0])
        if incoming:
            link_builder.set_joint_name(incoming['name'])
            parent_frame, child_frame = local_pose(incoming['local_pose0']), local_pose(incoming['local_pose1'])
            if incoming['axis']:
                alignment = sapien.Pose(q=axis_rotation[incoming['axis']])
                parent_frame, child_frame = parent_frame * alignment, child_frame * alignment
            limits = [float(x) for x in incoming['limits']]
            position_scale = scale[0] if incoming['type'] == 'prismatic' else 1
            if incoming['type'] == 'prismatic':
                limits = [x * position_scale for x in limits]
            link_builder.set_joint_properties(incoming['type'], [limits] if limits else [], parent_frame, child_frame)
            if incoming['type'] != 'fixed':
                initial_positions[incoming['name']] = incoming['initial_position'] * position_scale
                initial_velocities[incoming['name']] = incoming.get('initial_velocity', 0) * position_scale
        collision_meshes = [m for m in link['meshes'] if m['collision']]
        volume = sum(m['collision_volume'] for m in collision_meshes) * float(np.prod(scale))
        total_mass = effective_physics.get('mass')
        if total_mass is not None and len(links) > 1:
            if not disconnected or manifest['joints']:
                raise ValueError('A total articulation mass override needs an explicit per-link mass distribution')
            if all_volume <= 0:
                raise ValueError('Disconnected bodies need positive collision volume for the recorded mass allocation')
            total_mass = total_mass * volume / all_volume
        authored_mass = link['physics'].get('physics:mass')
        if authored_mass is not None:
            authored_mass *= kilograms
        authored_inertia = link['physics'].get('physics:diagonalInertia')
        if authored_inertia is not None and not all(float(x) > 0 for x in authored_inertia):
            authored_inertia = None
        if total_mass is None and authored_mass is not None and authored_mass > 0 and authored_inertia is None:
            total_mass = authored_mass
        if total_mass is not None:
            effective_masses[link['name']] = total_mass
        expected = 0
        for mesh in link['meshes']:
            if mesh['visual']:
                link_builder.add_visual_from_file(str(manifest_path.parent / mesh['geometry']), scale=scale)
            if not mesh['collision']:
                continue
            material = mesh.get('physics_material', {})
            static = effective_physics.get('static_friction', material.get('physics:staticFriction'))
            dynamic = effective_physics.get('dynamic_friction', material.get('physics:dynamicFriction'))
            restitution = effective_physics.get('restitution', material.get('physics:restitution'))
            collision_args = {'scale': scale}
            if all(x is not None for x in (static, dynamic, restitution)):
                collision_args['material'] = scene.create_physical_material(static, dynamic, restitution)
            density = link['physics'].get('physics:density', material.get('physics:density'))
            if density is not None:
                density *= kilograms / meters ** 3
            if total_mass is not None:
                if volume <= 0:
                    raise ValueError(f'Cannot preserve mass without positive collision volume: {link["name"]}')
                density = total_mass / volume
            if mesh['collision_type'] == 'triangle_mesh':
                if parent is not None or not fixed_root:
                    raise ValueError('Concave triangle collision is only valid on a fixed articulation root')
                for filename in mesh['collision_geometry']:
                    link_builder.add_nonconvex_collision_from_file(str(manifest_path.parent / filename), **collision_args)
                    expected += 1
            elif mesh['collision_type'] in ('convex_decomposition', 'source_convex_hull', 'source_convex_primitive'):
                if density is not None and density > 0:
                    collision_args['density'] = density
                for filename in mesh['collision_geometry']:
                    link_builder.add_convex_collision_from_file(str(manifest_path.parent / filename), **collision_args)
                    expected += 1
            else:
                raise ValueError(f'Unknown explicit collision conversion {mesh["collision_type"]}')
        if authored_mass is not None and authored_mass > 0 and authored_inertia is not None:
            factor = float(scale[0])
            if not np.allclose(scale, factor):
                raise ValueError('Nonuniform scale of authored inertia requires explicit inertia conversion')
            cmass = sapien.Pose(np.asarray(link['physics'].get('physics:centerOfMass', [0, 0, 0])) * meters * scale,
                                link['physics'].get('physics:principalAxes', [1, 0, 0, 0]))
            inertia = np.asarray(authored_inertia) * kilograms * meters ** 2 * factor ** 2
            link_builder.set_mass_and_inertia(total_mass if total_mass is not None else authored_mass, cmass, inertia)
            authored_inertia_links.add(link['name'])
        collision_counts[link['name']] = expected
        for joint in outgoing[path]:
            add_link(builder, joint['body1'], link_builder, joint)

    articulations = []
    for component in components:
        builder = scene.create_articulation_builder()
        add_link(builder, component)
        articulation = builder.build(fix_root_link=fixed_root)
        articulation.set_root_pose(pose * local_pose(links[component]['initial_pose']))
        active = articulation.get_active_joints()
        reset_to_targets = physics.get('reset_drive', False) and all(
            source_joints[joint.get_name()].get('drive') is not None for joint in active)
        qpos, qvel = [], []
        for joint in active:
            source = source_joints[joint.get_name()]
            initial = initial_positions[joint.get_name()]
            if reset_to_targets:
                initial = source['drive']['target_position'] * (scale[0] if source['type'] == 'prismatic' else 1)
                initial = float(np.clip(initial, *joint.get_limits()[0]))
            qpos.append(initial)
            qvel.append(initial_velocities[joint.get_name()])
        if qpos:
            articulation.set_qpos(qpos)
            articulation.set_qvel(qvel)
        for joint in active:
            source = source_joints[joint.get_name()]
            drive = source.get('drive')
            if drive is None:
                continue
            if drive['mode'] not in ('force', 'acceleration'):
                raise ValueError('Unrecognised source drive mode')
            if drive['mode'] == 'acceleration' and (source['type'] != 'prismatic' or meters != 1 or kilograms != 1):
                raise ValueError('Acceleration drive conversion is only verified for source SI prismatic joints')
            force_limit = min(float(drive['force_limit']), float(np.finfo(np.float32).max))
            joint.set_drive_properties(drive['stiffness'], drive['damping'], force_limit=force_limit, mode=drive['mode'])
            position_scale = scale[0] if source['type'] == 'prismatic' else 1
            joint.set_drive_target(drive['target_position'] * position_scale)
            joint.set_drive_velocity_target(drive['target_velocity'] * position_scale)
        articulations.append(articulation)
    if len(created) != len(links):
        raise ValueError('Some source rigid links were not built')
    loaded = ConvertedAsset(manifest, articulations, local_pose(links[root_path]['initial_pose']), (scale * meters).tolist())
    # This SAPIEN release silently skips failed shape cooking. Count actual shapes
    # so failed geometry cannot become an apparently successful empty asset.
    for articulation in articulations:
        for link in articulation.get_links():
            expected, actual = collision_counts[link.get_name()], len(link.get_collision_shapes())
            if actual != expected:
                raise RuntimeError(f'Collision cooking omitted shapes on {link.get_name()}: expected {expected}, got {actual}')
            source = next(item for item in links.values() if item['name'] == link.get_name())
            if link.get_name() in effective_masses:
                desired = effective_masses[link.get_name()]
                if link.get_name() not in authored_inertia_links:
                    # PhysX cooking can change convex volume, so mesh-derived
                    # density alone does not preserve the specified final mass.
                    link.inertia = np.asarray(link.inertia) * (desired / float(link.mass))
                link.mass = desired
            if 'physxRigidBody:disableGravity' in source['physics']:
                link.disable_gravity = source['physics']['physxRigidBody:disableGravity']
    loaded.effective_physics = effective_physics
    loaded.effective_mass_by_link = effective_masses
    loaded.fixed_root = fixed_root
    return loaded

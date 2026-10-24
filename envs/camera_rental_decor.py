"""Fixed rental-counter panels with one complete image on the front face."""

from pathlib import Path

import sapien


FRONT_PANEL_MESH = Path(__file__).parent / "scene_assets/camera_rental_front_panel.obj"


def create_front_image_panel(scene, pose, width, height, texture_id, name):
    """Create a static, visual-only X/Z plane facing the viewer at negative Y.

    The physical frame/backing is built separately by the scene. The four texture
    coordinates cover the image once, without stretching it over six box faces.
    """
    material = sapien.render.RenderMaterial()
    material.set_base_color_texture(sapien.render.RenderTexture2D(
        str(Path("assets/background_texture") / f"{texture_id}.png")
    ))
    material.base_color = [1, 1, 1, 1]
    material.metallic = 0.0
    material.roughness = 0.7
    builder = scene.create_actor_builder()
    builder.set_physx_body_type("static")
    builder.add_visual_from_file(
        filename=str(FRONT_PANEL_MESH), scale=[width, 1.0, height], material=material
    )
    panel = builder.build(name=name)
    panel.set_pose(pose)
    return panel

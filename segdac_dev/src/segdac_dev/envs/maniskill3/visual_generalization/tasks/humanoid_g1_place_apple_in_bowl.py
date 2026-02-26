import sapien
import numpy as np

from mani_skill.utils.registration import register_env
from mani_skill.envs.tasks.humanoid.humanoid_pick_place import UnitreeG1PlaceAppleInBowlEnv
from mani_skill.utils.scene_builder.kitchen_counter import KitchenCounterSceneBuilder
from transforms3d.euler import euler2quat


@register_env("UnitreeG1PlaceAppleInBowlTest-v1", max_episode_steps=100)
class UnitreeG1PlaceAppleInBowlTestEnv(UnitreeG1PlaceAppleInBowlEnv):
    def __init__(self, *args, **kwargs):
        self.test_config = kwargs.pop("test_config")
        super().__init__(*args, **kwargs)

    def _load_scene(self, options):
        self.scene_builder = KitchenCounterSceneBuilder(self)
        self.kitchen_scene = self.scene_builder.build(scale=self.kitchen_scene_scale)

        table_material_config = self.test_config["table"]["material"]
        table_texture = table_material_config["texture"]
        if table_texture is not None:
            table_texture = sapien.render.RenderTexture2D(table_texture)
        table_base_color = table_material_config["base_color"]
        if table_base_color is not None or table_texture is not None:
            for obj in self.scene_builder.table._objs:
                render_body_component = obj.find_component_by_type(sapien.render.RenderBodyComponent)
                for render_shape in render_body_component.render_shapes:
                    for part in render_shape.parts:
                        if table_base_color is not None:
                            part.material.set_base_color(table_base_color)
                            part.material.set_base_color_texture(None)
                        if table_texture is not None:
                            part.material.set_base_color_texture(table_texture)
                        part.material.set_normal_texture(None)
                        part.material.set_emission_texture(None)
                        part.material.set_transmission_texture(None)
                        part.material.set_metallic_texture(None)


        ground_material_config = self.test_config["ground"]["material"]
        ground_texture = ground_material_config["texture"]
        if ground_texture is not None:
            ground_texture = sapien.render.RenderTexture2D(ground_texture)
        ground_base_color = ground_material_config["base_color"]
        if ground_base_color is not None or ground_texture is not None:
            for obj in self.scene_builder.ground._objs:
                render_body_component = obj.find_component_by_type(sapien.render.RenderBodyComponent)
                for render_shape in render_body_component.render_shapes:
                    for part in render_shape.parts:
                        if ground_base_color is not None:
                            part.material.set_base_color(ground_base_color)
                            part.material.set_base_color_texture(None)
                        if ground_texture is not None:
                            part.material.set_base_color_texture(ground_texture)
                        part.material.set_normal_texture(None)
                        part.material.set_emission_texture(None)
                        part.material.set_transmission_texture(None)
                        part.material.set_metallic_texture(None)
                        part.material.set_roughness_texture(None)

        scale = self.kitchen_scene_scale
        builder = self.scene.create_actor_builder()
        fix_rotation_pose = sapien.Pose(q=euler2quat(np.pi / 2, 0, 0))
        ro_material_config = self.test_config["ro"]["material"]
        ro_base_color = ro_material_config["base_color"]
        ro_texture = ro_material_config["texture"]
        if ro_base_color is None and ro_texture is None:
            ro_material = None
        else:
            if ro_base_color is not None:
                ro_material = sapien.render.RenderMaterial(
                    base_color=ro_base_color,
                )
            else:
                ro_material = sapien.render.RenderMaterial()
                ro_material.base_color_texture = sapien.render.RenderTexture2D(ro_texture)
        builder.add_nonconvex_collision_from_file(
            filename="assets/bowl/frl_apartment_bowl_07.ply",
            pose=fix_rotation_pose,
            scale=[scale] * 3,
        )
        builder.add_visual_from_file(
            filename="assets/bowl/frl_apartment_bowl_07.glb",
            scale=[scale] * 3,
            pose=fix_rotation_pose,
            material=ro_material
        )
        builder.initial_pose = sapien.Pose(p=[0, -0.4, 0.753])
        self.bowl = builder.build_kinematic(name="bowl")

        builder = self.scene.create_actor_builder()
        mo_material_config = self.test_config["mo"]["material"]
        mo_base_color = mo_material_config["base_color"]
        mo_texture = mo_material_config["texture"]
        if mo_base_color is None and mo_texture is None:
            mo_material = None
        else:
            if mo_base_color is not None:
                mo_material = sapien.render.RenderMaterial(
                    base_color=mo_base_color,
                )
            else:
                mo_material = sapien.render.RenderMaterial()
                mo_material.base_color_texture = sapien.render.RenderTexture2D(mo_texture)

        builder.add_visual_from_file(
            filename="assets/apple/apple_1.glb",
            scale=[scale * 0.8] * 3,
            pose=fix_rotation_pose,
            material=mo_material
        )
        builder.add_multiple_convex_collisions_from_file(
            filename="assets/apple/apple_1.ply",
            pose=fix_rotation_pose,
            scale=[scale * 0.8] * 3,  # scale down more to make apple a bit smaller to be graspable
        )

        builder.initial_pose = sapien.Pose(p=[0, -0.4, 0.78])
        self.apple = builder.build(name="apple")

    def _load_lighting(self, options: dict):
        ambient_light_color = self.test_config["lighting"]["ambient_light_color"]
        if ambient_light_color is None:
            ambient_light_color = [0.3, 0.3, 0.3]
        self.scene.set_ambient_light(ambient_light_color)

        direction_light_1 = self.test_config["lighting"]["direction"]["light_1"]
        if direction_light_1 is None:
            direction_light_1 = [1, 1, -1]
        direction_light_2 = self.test_config["lighting"]["direction"]["light_2"]
        if direction_light_2 is None:
            direction_light_2 = [0, 0, -1]
        self.scene.add_directional_light(
            direction=direction_light_1,
            color=[1, 1, 1], 
            shadow=self.enable_shadow,
            position=[0, 0, 0], 
            shadow_scale=5,
            shadow_map_size=2048
        )
        self.scene.add_directional_light(
            direction=direction_light_2,
            color=[1, 1, 1],
            shadow=False,
            position=[0, 0, 0]
        )

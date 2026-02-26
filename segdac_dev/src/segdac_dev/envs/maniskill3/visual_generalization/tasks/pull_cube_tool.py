import sapien
import numpy as np

from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.envs.tasks.tabletop.pull_cube_tool import PullCubeToolEnv
from segdac_dev.envs.maniskill3.visual_generalization.tasks.scene_building import build_cube


@register_env("PullCubeToolTest-v1", max_episode_steps=100)
class PullCubeToolTestEnv(PullCubeToolEnv):
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02,**kwargs):
        self.test_config = kwargs.pop("test_config")
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=robot_init_qpos_noise, **kwargs)

    def _load_scene(self, options: dict):
        self.scene_builder = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()
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
                        part.material.set_roughness_texture(None)

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

        mo_material_config = self.test_config["mo"]["material"]
        mo_base_color = mo_material_config["base_color"]
        if mo_base_color is None:
            mo_base_color = np.array([255, 0, 0, 255]) / 255
        mo_material = sapien.render.RenderMaterial(
            base_color=mo_base_color,
        )
        if mo_material_config["texture"] is not None:
            mo_material.base_color_texture = sapien.render.RenderTexture2D(mo_material_config["texture"])

        self.l_shape_tool = self._build_l_shaped_tool(
            handle_length=self.handle_length,
            hook_length=self.hook_length,
            width=self.width,
            height=self.height,
            material=mo_material,
            visual_file_path=mo_material_config["visual_file_path"]
        )

        ro_material_config = self.test_config["ro"]["material"]
        ro_base_color = ro_material_config["base_color"]
        if ro_base_color is None:
            ro_base_color = np.array([12, 42, 160, 255]) / 255
        ro_material = sapien.render.RenderMaterial(
            base_color=ro_base_color,
        )
        if ro_material_config["texture"] is not None:
            ro_material.base_color_texture = sapien.render.RenderTexture2D(ro_material_config["texture"])

        self.cube = build_cube(
            scene=self.scene,
            half_size=self.cube_half_size,
            material=ro_material,
            visual_file_path=ro_material_config["visual_file_path"],
            name="cube",
            body_type="dynamic",
        )

    def _build_l_shaped_tool(self, handle_length, hook_length, width, height, material: sapien.render.RenderMaterial, visual_file_path: str):
        builder = self.scene.create_actor_builder()

        builder.add_box_collision(
            sapien.Pose([handle_length / 2, 0, 0]),
            [handle_length / 2, width / 2, height / 2],
            density=500,
        )
        builder.add_box_collision(
            sapien.Pose([handle_length - hook_length / 2, width, 0]),
            [hook_length / 2, width, height / 2],
        )

        if visual_file_path is not None:
            builder.add_visual_from_file(
                filename=visual_file_path,
                material=material,
                scale=[1.0, 1.0, 1.0]
            )
        else:
            builder.add_box_visual(
                sapien.Pose([handle_length / 2, 0, 0]),
                [handle_length / 2, width / 2, height / 2],
                material=material,
            )
            builder.add_box_visual(
                sapien.Pose([handle_length - hook_length / 2, width, 0]),
                [hook_length / 2, width, height / 2],
                material=material,
            )

        return builder.build(name="l_shape_tool")

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

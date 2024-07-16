import os
import sys
import bpy

sys.path.append(os.path.dirname(bpy.data.filepath))

from blender.render_script import Renderer  # noqa

# ------------------------------------------------------------------------------------- #

renderer = Renderer()


class SelectFolderOperator(bpy.types.Operator):
    bl_idname = "object.select_folder"
    bl_label = "Select Folder"

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        selected_path = context.window_manager.fileselect_add
        abs_path = bpy.path.abspath(selected_path)
        context.scene.selected_path = abs_path
        self.report({"INFO"}, f"Selected path: {abs_path}")
        return {"FINISHED"}


class LoadSampleOperator(bpy.types.Operator):
    bl_idname = "object.load_sample"
    bl_label = "Load Sample"

    def execute(self, context):
        selected_path = context.scene.selected_path
        selected_id = context.scene.selected_id
        selected_rate = context.scene.selected_rate
        render_mode = bpy.context.scene.render_mode.lower()
        self.report(
            {"INFO"}, f"Selected path: {selected_path}, Selected ID: {selected_id}"
        )
        renderer.render_cli(
            selected_path.replace("//", ""), selected_id, selected_rate, render_mode
        )
        return {"FINISHED"}


# ------------------------------------------------------------------------------------- #


class LOADING_PT_load_sample_panel(bpy.types.Panel):
    bl_label = "E.T. Sample Loading"
    bl_idname = "LOADING_PT_load_sample_panel"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_category = "Object"

    def draw(self, context):
        layout = self.layout

        # Add a text label
        layout.label(text="Path:")
        # Add a text input field to store the path
        layout.prop(context.scene, "selected_path", text="")
        # Add a text label
        layout.label(text="Sample ID:")
        # Add a text input field to store the sample ID
        layout.prop(context.scene, "selected_id", text="")
        # Add a text label
        layout.label(text="Rate of poses:")
        # Add a text input field to store the rate of poses
        layout.prop(context.scene, "selected_rate", text="")
        # Add a button to select the render mode
        layout.prop(context.scene, "render_mode")
        # Add a button to load the sample
        layout.operator("object.load_sample", text="Load sample")


# ------------------------------------------------------------------------------------- #


def register():
    bpy.utils.register_class(LOADING_PT_load_sample_panel)
    bpy.utils.register_class(SelectFolderOperator)
    bpy.utils.register_class(LoadSampleOperator)
    bpy.types.Scene.selected_path = bpy.props.StringProperty(
        name="Sample ID",
        default="//et-data/",
        subtype="DIR_PATH",
    )
    bpy.types.Scene.selected_id = bpy.props.StringProperty(
        name="Sample ID", default="2011_bmgWabhgTyI_00006_00000"
    )
    bpy.types.Scene.selected_rate = bpy.props.FloatProperty(
        name="Rate of poses", default=0.2
    )
    bpy.types.Scene.render_mode = bpy.props.EnumProperty(
        name="Generation Mode",
        description="Choose generation mode",
        items=[
            ("IMAGE", "Image", ""),
            ("VIDEO", "Video", ""),
            ("VIDEO_ACCUMULATE", "Video Accumulate", ""),
        ],
        default="IMAGE",
    )


def unregister():
    bpy.utils.unregister_class(LOADING_PT_load_sample_panel)
    bpy.utils.unregister_class(SelectFolderOperator)
    bpy.utils.unregister_class(LoadSampleOperator)
    del bpy.types.Scene.selected_path
    del bpy.types.Scene.selected_id
    del bpy.types.Scene.render_mode


# ------------------------------------------------------------------------------------- #

if __name__ == "__main__":
    register()

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
import os
from .segment import segment_pcd
from .register import register
from .extract import extract_plane
from .settings import Settings


class AppWindow:
    MENU_OPEN = 1
    MENU_EXPORT = 2
    MENU_QUIT = 3

    MENU_SEGMENT = 5
    MENU_REGISTER = 6
    MENU_EXTRACT = 7

    MENU_SHOW_SETTINGS = 11
    MENU_ABOUT = 21

    def __init__(self, vfov=60, max_pcd_vertices=1 << 20, **callbacks):
        self.vfov = vfov
        self.max_pcd_vertices = max_pcd_vertices
        self.settings = Settings()
        self.callbacks = callbacks
        # initialize the global application instance
        gui.Application.instance.initialize()
        # ---- Create main window ----
        self.window = gui.Application.instance.create_window(
            "Surgical Planning Software || Amputation", 800, 600)
        w = self.window
        em = w.theme.font_size
        # ---- Rendering scene ----
        self._scene = gui.SceneWidget()
        self._scene.enable_scene_caching(True)
        self._scene.scene = rendering.Open3DScene(w.renderer)
        self.pcd_bounds = o3d.geometry.AxisAlignedBoundingBox([-3, -3, 0],
                                                              [3, 3, 6])
        self.geometry = None
        self.camera_view()
        self.coor_base = None
        self.flag_exit = False
        self.flag_gui_init = False

        # ---- Menu ----
        if gui.Application.instance.menubar is None:
            file_menu = gui.Menu()
            file_menu.add_item("Open...", AppWindow.MENU_OPEN)
            file_menu.add_item("Export Current pcd...", AppWindow.MENU_EXPORT)
            file_menu.add_separator()
            file_menu.add_item("Quit", AppWindow.MENU_QUIT)

            func_menu = gui.Menu()
            func_menu.add_item("Segment Current pcd", AppWindow.MENU_SEGMENT)
            func_menu.add_item("Register based pose", AppWindow.MENU_REGISTER)
            func_menu.add_item("Extract Amputation plane", AppWindow.MENU_EXTRACT)

            settings_menu = gui.Menu()
            settings_menu.add_item("Settings",
                                   AppWindow.MENU_SHOW_SETTINGS)
            settings_menu.set_checked(AppWindow.MENU_SHOW_SETTINGS, True)

            help_menu = gui.Menu()
            help_menu.add_item("About", AppWindow.MENU_ABOUT)

            menu = gui.Menu()
            menu.add_menu("File", file_menu)
            menu.add_menu('Function', func_menu)
            menu.add_menu("Settings", settings_menu)
            menu.add_menu("Help", help_menu)
            gui.Application.instance.menubar = menu

        # connect the menu items to the window
        w.set_on_menu_item_activated(AppWindow.MENU_OPEN, self._on_menu_open)
        w.set_on_menu_item_activated(AppWindow.MENU_EXPORT, self._on_menu_export)
        w.set_on_menu_item_activated(AppWindow.MENU_QUIT, self._on_menu_quit)
        w.set_on_menu_item_activated(AppWindow.MENU_SEGMENT, self._on_menu_segment)
        w.set_on_menu_item_activated(AppWindow.MENU_REGISTER, self._on_menu_register)
        w.set_on_menu_item_activated(AppWindow.MENU_EXTRACT, self._on_menu_extract)

        w.set_on_menu_item_activated(AppWindow.MENU_SHOW_SETTINGS,
                                     self._on_menu_toggle_settings_panel)
        w.set_on_menu_item_activated(AppWindow.MENU_ABOUT, self._on_menu_about)
        # ----

        # ---- Setting Panel ----
        self._settings_panel = gui.Vert(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        tabs = gui.TabControl()
        # tab register
        tab1 = gui.Vert()
        tab1_1 = gui.VGrid(3, 0.5 * em, gui.Margins(em, em, em, em))
        tab1_1.add_child(gui.Label('pose file:'))
        self._file_edit = gui.TextEdit()
        tab1_1.add_child(self._file_edit)
        file_button = gui.Button("...")
        file_button.horizontal_padding_em = 0.5
        file_button.vertical_padding_em = 0
        file_button.set_on_clicked(self._on_file_button)
        tab1_1.add_child(file_button)
        tab1_1.add_child(gui.Label('pcds floder:'))
        self._folder_edit = gui.TextEdit()
        tab1_1.add_child(self._folder_edit)
        folder_button = gui.Button("...")
        folder_button.horizontal_padding_em = 0.5
        folder_button.vertical_padding_em = 0
        folder_button.set_on_clicked(self._on_folder_button)
        tab1_1.add_child(folder_button)
        reg_button = gui.Button("Register")
        reg_button.horizontal_padding_em = 0.5
        reg_button.vertical_padding_em = 0.1
        reg_button.set_on_clicked(self._on_menu_register)
        tab1.add_child(tab1_1)
        tab1.add_child(reg_button)
        # tab Segmentation
        tab2 = gui.Vert()
        tab2_1 = gui.VGrid(2, 0.25*em)
        # Pass through
        pass_through = gui.Checkbox("Pass Through")
        pass_through.checked = True
        pass_through.set_on_checked(self._on_check_pass_through)
        pass_vec_edit = gui.VectorEdit()
        pass_vec_edit.vector_value = [1000, 1000, 1200]
        pass_vec_edit.set_on_value_changed(self._on_edit_pass_vec)
        tab2_1.add_child(pass_through)
        tab2_1.add_child(pass_vec_edit)
        # noise reduct
        noise_reduct = gui.Checkbox("Noise Reduct")
        noise_reduct.checked = True
        noise_reduct.set_on_checked(self._on_check_noise_reduct)
        tab2_1_1 = gui.Horiz()
        noise_reduct_nb = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        noise_reduct_nb.int_value = 100
        noise_reduct_nb.set_on_value_changed(self._on_edit_noise_reduct_nb)
        noise_reduct_std = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        noise_reduct_std.int_value = 1
        noise_reduct_std.set_on_value_changed(self._on_edit_noise_reduct_std)
        tab2_1_1.add_child(gui.Label("k:  "))
        tab2_1_1.add_child(noise_reduct_nb)
        tab2_1_1.add_stretch()
        tab2_1_1.add_child(gui.Label("std: "))
        tab2_1_1.add_child(noise_reduct_std)
        tab2_1.add_child(noise_reduct)
        tab2_1.add_child(tab2_1_1)
        # down sample
        down_sample = gui.Checkbox("Down Sample")
        down_sample.checked = True
        down_sample.set_on_checked(self._on_check_down_sample)
        tab2_1_2 = gui.Horiz()
        down_sample_size = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        down_sample_size.int_value = 1
        down_sample_size.set_on_value_changed(self._on_edit_down_sample_size)
        down_sample_pts = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        down_sample_pts.int_value = 50000
        down_sample_pts.set_on_value_changed(self._on_edit_down_sample_pts)
        tab2_1_2.add_child(gui.Label("size:"))
        tab2_1_2.add_child(down_sample_size)
        tab2_1_2.add_stretch()
        tab2_1_2.add_child(gui.Label("n:"))
        tab2_1_2.add_child(down_sample_pts)
        tab2_1.add_child(down_sample)
        tab2_1.add_child(tab2_1_2)
        # remove plane
        remove_plane = gui.Checkbox("Remove Plane")
        remove_plane.checked = True
        remove_plane.set_on_checked(self._on_check_remove_plane)
        tab2_1_3 = gui.Horiz()
        remove_plane_d = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        remove_plane_d.int_value = 10
        remove_plane_d.set_on_value_changed(self._on_edit_remove_plane_d)
        remove_plane_minpts = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        remove_plane_minpts.int_value = 3
        remove_plane_minpts.set_on_value_changed(self._on_edit_remove_plane_minpts)
        tab2_1_3.add_child(gui.Label("d:   "))
        tab2_1_3.add_child(remove_plane_d)
        tab2_1_3.add_stretch()
        tab2_1_3.add_child(gui.Label("n:"))
        tab2_1_3.add_child(remove_plane_minpts)
        tab2_1.add_child(remove_plane)
        tab2_1.add_child(tab2_1_3)
        # dbscan segmentation
        dbscan_seg = gui.Checkbox("Segment")
        dbscan_seg.checked = True
        dbscan_seg.set_on_checked(self._on_check_dbscan_seg)
        tab2_1_4 = gui.Horiz()
        dbscan_seg_eps = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        dbscan_seg_eps.int_value = 10
        dbscan_seg_eps.set_on_value_changed(self._on_edit_dbscan_seg_eps)
        dbscan_seg_minpts = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        dbscan_seg_minpts.int_value = 10
        dbscan_seg_minpts.set_on_value_changed(self._on_edit_dbscan_seg_minpts)
        tab2_1_4.add_child(gui.Label("eps: "))
        tab2_1_4.add_child(dbscan_seg_eps)
        tab2_1_4.add_stretch()
        tab2_1_4.add_child(gui.Label("n:"))
        tab2_1_4.add_child(dbscan_seg_minpts)
        tab2_1.add_child(dbscan_seg)
        tab2_1.add_child(tab2_1_4)
        seg_button = gui.Button("Segment")
        seg_button.horizontal_padding_em = 0.5
        seg_button.vertical_padding_em = 0.1
        seg_button.set_on_clicked(self._on_menu_segment)
        tab2_1.add_child(seg_button)
        tab2.add_child(tab2_1)
        # tab Extract
        tab3 = gui.Vert()
        tab3_1 = gui.VGrid(2, 0.5 * em, gui.Margins(em, em, em, em))
        tab3_1.add_child(gui.Label('Distance from knee(mm):'))
        dis_from_knee = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        dis_from_knee.double_value = 100
        dis_from_knee.set_on_value_changed(self._on_edit_dis_from_knee)
        tab3_1.add_child(dis_from_knee)
        tab3_1.add_child(gui.Label('Length of instrucment(mm):'))
        len_instrucment = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        len_instrucment.double_value = 400
        len_instrucment.set_on_value_changed(self._on_edit_len_instrucment)
        tab3_1.add_child(len_instrucment)

        extract_plane_button = gui.Button("Extract")
        extract_plane_button.horizontal_padding_em = 0.5
        extract_plane_button.vertical_padding_em = 0.1
        extract_plane_button.set_on_clicked(self._on_menu_extract)
        save_coors_button = gui.Button("Save coors")
        save_coors_button.horizontal_padding_em = 0.5
        save_coors_button.vertical_padding_em = 0.1
        save_coors_button.set_on_clicked(self._on_button_save)
        tab3_2 = gui.Horiz()
        tab3_2.add_stretch()
        tab3_2.add_child(extract_plane_button)
        tab3_2.add_fixed(em)
        tab3_2.add_child(save_coors_button)
        tab3_2.add_stretch()
        tab3.add_child(tab3_1)
        tab3.add_child(tab3_2)

        tabs.add_tab('Segment', tab2)
        tabs.add_tab('Register', tab1)
        tabs.add_tab('Extract', tab3)
        self._show_processing = gui.CollapsableVert("Processing")
        self._show_processing.set_is_open(False)
        self._show_processing.add_child(tabs)
        self._settings_panel.add_child(self._show_processing)
        self._show_axes = gui.Checkbox("Show axes")
        self._show_axes.set_on_checked(self._on_show_axes)
        self._settings_panel.add_child(self._show_axes)

        self._save_pcd_button = gui.Button("Save PCD")
        self._save_pcd_button.horizontal_padding_em = 0.5
        self._save_pcd_button.vertical_padding_em = 0.1
        self._toggle_camera = gui.ToggleSwitch("Camera On/Off")
        self._toggle_camera.is_on = False
        if 'on_toggle_camera' in callbacks:
            self._toggle_camera.set_on_clicked(callbacks['on_toggle_camera'])
        else:
            self._toggle_camera.enabled = False
        save_buttons = gui.Horiz(em)
        save_buttons.add_stretch()
        save_buttons.add_child(self._toggle_camera)
        save_buttons.add_stretch()
        save_buttons.add_child(self._save_pcd_button)
        save_buttons.add_stretch()
        self._settings_panel.add_child(save_buttons)
        # show video
        self.video_size = (int(240 * self.window.scaling),
                           int(320 * self.window.scaling), 3)
        self.show_color = gui.CollapsableVert("Color image")
        self.show_color.set_is_open(False)
        self._settings_panel.add_child(self.show_color)
        self.color_video = gui.ImageWidget(
            o3d.geometry.Image(np.zeros(self.video_size, dtype=np.uint8)))
        self.show_color.add_child(self.color_video)
        self.show_depth = gui.CollapsableVert("Depth image")
        self.show_depth.set_is_open(False)
        self._settings_panel.add_child(self.show_depth)
        self.depth_video = gui.ImageWidget(
            o3d.geometry.Image(np.zeros(self.video_size, dtype=np.uint8)))
        self.show_depth.add_child(self.depth_video)
        self.status_message = gui.Label("")
        self._settings_panel.add_child(self.status_message)
        # ----

        # ---- Layout callback ----
        w.set_on_layout(self._on_layout)
        if 'on_window_close' in callbacks:
            w.set_on_close(callbacks['on_window_close'])
        w.add_child(self._scene)
        w.add_child(self._settings_panel)
        # ----

        self._apply_settings()

    def _apply_settings(self):
        self._scene.scene.show_axes(self.settings.show_axes)
        self._show_axes.checked = self.settings.show_axes
        if self._toggle_camera.is_on:
            self._save_pcd_button.set_on_clicked(self.callbacks['on_save_pcd'])
        else:
            self._save_pcd_button.set_on_clicked(self._on_menu_export)

    def camera_view(self):
        """Callback to reset point cloud view to the camera"""
        self._scene.scene.clear_geometry()
        if self.geometry is not None:
            try:
                # Point cloud
                self._scene.scene.add_geometry("__model__", self.geometry,
                                               self.settings.material)
                bounds = self._scene.scene.bounding_box
                self._scene.setup_camera(self.vfov, bounds, bounds.get_center())
                self._scene.scene.camera.look_at([0, 0, 0], [0, 0, -1], [0, -1, 0])  # center eye up
            except Exception as e:
                print(e)
        else:
            self._scene.setup_camera(self.vfov, self.pcd_bounds, [0, 0, 0])
            self._scene.scene.camera.look_at([0, 0, 0], [0, 0, -1], [0, -1, 0])

    def _on_layout(self, layout_context):
        #   在on_layout回调函数中应正确设置所有子对象的框架(position + size)，
        #   回调结束之后才会布局孙子对象。
        r = self.window.content_rect
        self._scene.frame = r
        pannel_width = 24 * layout_context.theme.font_size
        pannel_height = min(
            r.height, self._settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height
        )
        self._settings_panel.frame = gui.Rect(r.get_right() - pannel_width, r.y, pannel_width, pannel_height)
        # print(r.x, r.y)

    def _on_menu_open(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose file to load",
                             self.window.theme)

        dlg.add_filter("", "All files")
        dlg.set_path('./')

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_dialog_done)
        self.window.show_dialog(dlg)

    def _on_file_dialog_cancel(self):
        self.window.close_dialog()

    def _on_load_dialog_done(self, filename):
        self.window.close_dialog()
        self.load(filename)

    def _on_menu_export(self):
        if not self.geometry :
            self._message_box('Error', 'Missing pointcloud!')
            return
        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save",
                             self.window.theme)
        dlg.set_path('./')
        dlg.add_filter(".pcd .ply", "PCD files")
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_export_dialog_done)
        self.window.show_dialog(dlg)

    def _on_export_dialog_done(self, filename):
        self.window.close_dialog()
        pcd = self.geometry
        o3d.io.write_point_cloud(filename, pcd)
        self.status_message.text = "Point cloud saved: " + filename

    def _on_menu_quit(self):
        gui.Application.instance.quit()

    def _on_menu_segment(self):
        if not self.geometry:
            self._message_box('Error', 'Missing pointcloud!')
            return
        # segment the leg out of pointcloud
        self.geometry = segment_pcd(self.geometry, self.settings)
        self.camera_view()
        self.status_message.text = "Point cloud segmented."

    def _on_menu_register(self):
        if not all([self._file_edit.text_value, self._folder_edit.text_value]):
            self._message_box('Error', 'Filling in paths above!')
            return
        pose = self._file_edit.text_value
        pcds = self._folder_edit.text_value
        pcds_list = os.listdir(pcds)
        # wrong file type
        if not pose.endswith(('.txt', '.npy')):
            self._message_box('Error', 'pose file should be .npy!')
            return
        for i in pcds_list:
            if not i.endswith('.pcd'):
                self._message_box('Error', 'pcd files should be .pcd!')
                return
        self.geometry = register(pose,pcds)
        self.camera_view()
        self.status_message.text = "Point clouds registered."
    def _on_menu_extract(self):
        if not all([self._file_edit.text_value, self.geometry]):
            self._message_box('Error', 'Missing pointcloud or filepath!')
            return
        pose = self._file_edit.text_value
        leg_pcd, plane_pcd, result = extract_plane(self.geometry, pose,
                                                   self.settings.dis_from_knee,
                                                   self.settings.len_instrucment)
        self.coor_base = result
        self.geometry = leg_pcd+plane_pcd
        self.camera_view()
        self.status_message.text = "Amputation plane extracted."
    def _on_button_save(self):
        if self.coor_base is None:
            self._message_box('Error', 'Extract and Calculate coordinates first!')
            return
        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Save Coordinates based robot base?",
                             self.window.theme)
        dlg.set_path('./')
        dlg.add_filter(".txt", "Coordinate (.txt)")
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_save_result)
        self.window.show_dialog(dlg)
    def _on_save_result(self, filename):
        self.window.close_dialog()
        np.savetxt(filename, self.coor_base, '%.2f')
        self.status_message.text = "Coordinate file saved: " + filename
    def _on_file_button(self):
        # select pose file
        filedlg = gui.FileDialog(gui.FileDialog.OPEN, "Select file",
                                 self.window.theme)
        filedlg.add_filter(".npy .txt", "pose file(.npy .txt)")
        filedlg.add_filter("", "All files")
        filedlg.set_on_cancel(self._on_file_dialog_cancel)
        filedlg.set_on_done(self._load_pose_file)
        self.window.show_dialog(filedlg)
    def _load_pose_file(self, filename):
        self.window.close_dialog()
        self._file_edit.text_value = filename
        self.status_message.text = "Pose file loaded: "+filename

    def _on_folder_button(self):
        # select pose file
        filedlg = gui.FileDialog(gui.FileDialog.OPEN_DIR, "Select folder",
                                 self.window.theme)
        filedlg.set_on_cancel(self._on_file_dialog_cancel)
        filedlg.set_on_done(self._load_pcds_file)
        self.window.show_dialog(filedlg)
    def _load_pcds_file(self, folder_name):
        self.window.close_dialog()
        self._folder_edit.text_value = folder_name


    def _on_menu_toggle_settings_panel(self):
        self._settings_panel.visible = not self._settings_panel.visible
        gui.Application.instance.menubar.set_checked(
            AppWindow.MENU_SHOW_SETTINGS, self._settings_panel.visible)

    def _message_box(self, title, label):
        em = self.window.theme.font_size
        dlg = gui.Dialog(title)
        # Add the text
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label(label))
        # Add the Ok button. We need to define a callback function to handle
        # the click.
        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)
        # We want the Ok button to be the right side, so we need to add
        # a stretch item to the layout, otherwise the button will be the size
        # of the entire row. A stretch item takes up as much space as it can,
        # which forces the button to be its minimum size.
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)
    def _on_menu_about(self):
        self._message_box('About', '3D reconstruction software')

    def _on_about_ok(self):
        self.window.close_dialog()
    def _on_show_axes(self, show):
        self.settings.show_axes = show
        self._apply_settings()
        self.status_message.text = 'Show axes' + (' On' if show else ' Off')
    def _on_check_pass_through(self, pass_through):
        self.settings.pass_through = pass_through
        self.status_message.text = 'Pass through'+(' On' if pass_through else ' Off')
    def _on_edit_pass_vec(self, vector):
        self.settings.pass_x = vector[0]
        self.settings.pass_y = vector[1]
        self.settings.pass_z = vector[2]
    def _on_check_noise_reduct(self, noise_reduct):
        self.settings.noise_reduct = noise_reduct
        self.status_message.text = 'Noise reduction' + (' On' if noise_reduct else ' Off')
    def _on_edit_noise_reduct_nb(self, nb):
        self.settings.noise_nb_neighbors = nb
    def _on_edit_noise_reduct_std(self, nb):
        self.settings.noise_std_radio = nb
    def _on_check_down_sample(self, down_sample):
        self.settings.down_sample = down_sample
        self.status_message.text = 'Down sample' + (' On' if down_sample else ' Off')
    def _on_edit_down_sample_size(self, size):
        self.settings.down_sample_size = size
    def _on_edit_down_sample_pts(self, pts):
        self.settings.down_sample_points = pts
    def _on_check_remove_plane(self, remove_plane):
        self.settings.rm_plane = remove_plane
        self.status_message.text = 'Remove plane' + (' On' if remove_plane else ' Off')
    def _on_edit_remove_plane_d(self, d):
        self.settings.rm_plane_d = d
    def _on_edit_remove_plane_minpts(self, pts):
        self.settings.rm_plane_min_points = pts
    def _on_check_dbscan_seg(self, seg):
        self.settings.dbscan_seg = seg
        self.status_message.text = 'DBSCAN segmentation' + (' On' if seg else ' Off')
    def _on_edit_dbscan_seg_eps(self, eps):
        self.settings.dbscan_seg_eps = eps
    def _on_edit_dbscan_seg_minpts(self, pts):
        self.settings.dbscan_seg_min_points = pts
    def _on_edit_dis_from_knee(self, dis):
        self.settings.dis_from_knee = dis
    def _on_edit_len_instrucment(self, len):
        self.settings.len_instrucment = len

    def load(self, path):
        if os.path.isfile(path):
            self.geometry = None

            print("[Info]", path, "appears to be a point cloud")
            cloud = None
            try:
                cloud = o3d.io.read_point_cloud(path)
            except Exception:
                pass
            if cloud is not None:
                print("[Info] Successfully read", path)
                self.status_message.text = "[Info] Successfully read", path
                if not cloud.has_normals():
                    cloud.estimate_normals()
                cloud.normalize_normals()
                unit = 1000  # 1000 for mm, 1 for m
                cloud.transform([[unit, 0, 0, 0], [0, unit, 0, 0], [0, 0, unit, 0], [0, 0, 0, 1]])
                self.geometry = cloud
            else:
                print("[WARNING] Failed to read points", path)
                self.status_message.text = "[WARNING] Failed to read points", path
            self.camera_view()

    def update(self, frame_elements):
        """Update visualization with point cloud and images. Must run in main
                thread since this makes GUI calls.

                Args:
                    frame_elements: dict {element_type: geometry element}.
                        Dictionary of element types to geometry elements to be updated
                        in the GUI:
                            'raw': point cloud,
                            'color': rgb image (3 channel, uint8),
                            'depth': depth image (uint8),
                            'status_message': message
                """
        if not self.flag_gui_init:
            # Set dummy point cloud to allocate graphics memory
            dummy_pcd = o3d.t.geometry.PointCloud({
                'positions':
                    o3d.core.Tensor.zeros((self.max_pcd_vertices, 3),
                                          o3d.core.Dtype.Float32),
                'colors':
                    o3d.core.Tensor.zeros((self.max_pcd_vertices, 3),
                                          o3d.core.Dtype.Float32),
                'normals':
                    o3d.core.Tensor.zeros((self.max_pcd_vertices, 3),
                                          o3d.core.Dtype.Float32)
            })
            if self._scene.scene.has_geometry('raw'):
                self._scene.scene.remove_geometry('raw')

            self._scene.scene.add_geometry('raw', dummy_pcd, self.settings.material)
            self._scene.scene.camera.look_at([0, 0, 0], [0, 0, -1], [0, -1, 0])
            self.flag_gui_init = True

        if os.name == 'nt':
            self._scene.scene.remove_geometry('raw')
            self._scene.scene.add_geometry('raw', frame_elements['raw'],
                                            self.settings.material)
        else:
            update_flags = (rendering.Scene.UPDATE_POINTS_FLAG |
                            rendering.Scene.UPDATE_COLORS_FLAG)
            self._scene.scene.scene.update_geometry('raw',
                                                     frame_elements['raw'],
                                                     update_flags)
        # Update color and depth images
        if self.show_color.get_is_open() and 'color' in frame_elements:
            sampling_ratio = self.video_size[1] / frame_elements['color'].columns
            self.color_video.update_image(
                frame_elements['color'].resize(sampling_ratio).cpu())
        if self.show_depth.get_is_open() and 'depth' in frame_elements:
            sampling_ratio = self.video_size[1] / frame_elements['depth'].columns
            self.depth_video.update_image(
                frame_elements['depth'].resize(sampling_ratio).cpu())
        if 'status_message' in frame_elements:
            self.status_message.text = frame_elements["status_message"]
        self._scene.force_redraw()

    def run(self):
        gui.Application.instance.run()


if __name__ == "__main__":
    app = AppWindow()
    app.run()
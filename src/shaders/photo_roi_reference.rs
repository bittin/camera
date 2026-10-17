// SPDX-License-Identifier: GPL-3.0-only
//! Test-only immutable full-sensor 16x16 oracle.
use super::*;
const REFERENCE_SHADER: &str = concat!(
    include_str!("roi_reference_shader/debayer.wgsl"),
    "\n",
    include_str!("roi_reference_shader/debayer_texture_functions.wgsl"),
    "\n",
    include_str!("roi_reference_shader/bayer_finish_functions.wgsl"),
    "\n",
    include_str!("roi_reference_shader/debayer_photo.wgsl")
);
impl GpuConvertPipeline {
    pub(crate) fn install_roi_frozen_reference(&mut self) {
        self.ensure_photo_bilinear_pipeline();
        // Frozen shader is 16x16; candidate constructor retains production 64x4.
        self.photo_workgroup_size = [16, 16];
        self.photo_bilinear_pipeline = Some(self.create_pipeline(
            REFERENCE_SHADER,
            "immutable baseline photo_bilinear",
            &[
                (0, BindingSpec::Texture),
                (1, BindingSpec::StorageTexture),
                (2, BindingSpec::Uniform),
                (3, BindingSpec::ReadOnlyStorageBuffer),
                (4, BindingSpec::Texture),
            ],
        ));
    }
}

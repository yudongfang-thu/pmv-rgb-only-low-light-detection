"""xm_full_r1: ultralytics default training augmentation (RandomHSV + Mosaic, close_mosaic 10) on the paired RGB-IR loader.

The frozen paired-loader contract (paired_rgbir_data.validate_frozen_geometry) rejects RandomHSV and Mosaic. Its IR
stream is the RGB transform replayed with the same RNG: HSV replays as is, but Mosaic would take its three extra tiles
from the RGB dataset. Two changes, nothing else in the loader:
  allow_full_augmentation  the contract accepts RandomHSV at exactly the recipe gains and Mosaic (n = 4) at exactly the
                           recipe probability or 0 (after close_mosaic); every other rule is checked as before, with
                           those members neutralised during the original check
  install_paired_mosaic    in the IR replay of a mosaic sample the tiles are the IR counterparts of the RGB tiles: the
                           RGB pass records the tile indexes; the replay still calls the original get_indexes (same RNG
                           consumption) but uses the recorded indexes and reads the tiles through the explicit RGB -> IR
                           mapping (equal original shapes asserted). close_mosaic rebuilds the RGB transforms; the
                           contract is re-checked and the tracked dataset's geometry taps are re-installed.
The IR stream receives the same HSV draw as RGB (a grey 3-channel IR image only changes in V), as in xm_hsv_r1.
Under mosaic the tracked per-sample geometry matrix is not mosaic-aware; it stays identical for RGB and IR and is not
read by the RGBU / PVC / PVC_AUG objectives.
"""
from native_recipe import RECIPE

OBSERVED = dict(hsv_gains=[], mosaic_p=[], mosaic_samples=0, close_mosaic_calls=0)


def _members(transform):
    import paired_rgbir_data as prd
    return list(prd._transform_members(transform))


def active_mosaic(transform):
    """The enabled Mosaic member of a pipeline, or None (disabled, e.g. after close_mosaic)."""
    active = [m for m in _members(transform) if type(m).__name__ == 'Mosaic' and float(m.p) > 0.]
    if len(active) > 1:
        raise RuntimeError('More than one enabled Mosaic in the pipeline')
    return active[0] if active else None


def allow_full_augmentation():
    import paired_rgbir_data as prd
    if getattr(prd.validate_frozen_geometry, 'xm_full_r1', False):
        return
    aug = RECIPE['augmentation']
    want_hsv = (float(aug['hsv_h']), float(aug['hsv_s']), float(aug['hsv_v']))
    want_mosaic = float(aug['mosaic'])
    original = prd.validate_frozen_geometry

    def validate(transform):
        members = _members(transform)
        hsv = [m for m in members if type(m).__name__ == 'RandomHSV']
        mosaic = [m for m in members if type(m).__name__ == 'Mosaic']
        for m in hsv:
            got = (float(m.hgain), float(m.sgain), float(m.vgain))
            if got != want_hsv:
                raise prd.PairingContractError('RandomHSV gains differ from the xm_full_r1 recipe: %s' % (got,))
            OBSERVED['hsv_gains'].append(list(got))
        for m in mosaic:
            if float(m.p) not in (want_mosaic, 0.) or int(getattr(m, 'n', -1)) != 4:
                raise prd.PairingContractError('Mosaic differs from the xm_full_r1 recipe: p=%s n=%s' % (m.p, getattr(m, 'n', None)))
            OBSERVED['mosaic_p'].append(float(m.p))
        saved_hsv = [(m, m.hgain, m.sgain, m.vgain) for m in hsv]
        saved_p = [(m, m.p) for m in mosaic]
        try:
            for m in hsv:
                m.hgain = m.sgain = m.vgain = 0.
            for m in mosaic:
                m.p = 0.
            original(transform)  # every other rule of the frozen contract still applies
        finally:
            for m, h, s, v in saved_hsv:
                m.hgain, m.sgain, m.vgain = h, s, v
            for m, p in saved_p:
                m.p = p

    validate.xm_full_r1 = True
    prd.validate_frozen_geometry = validate


class IRTiles:
    """Mosaic's dataset during the IR replay: tile i is the IR counterpart of RGB record i."""

    def __init__(self, paired):
        self.paired = paired

    def __len__(self):
        return len(self.paired.base)

    @property
    def buffer(self):
        return self.paired.base.buffer

    def get_image_and_label(self, index):
        import paired_rgbir_data as prd
        base = self.paired.base
        strong = self.paired._mapping[prd._canonical(base.im_files[index])]
        record = self.paired._teacher_record(strong)
        if tuple(record['ori_shape']) != tuple(base.labels[index]['shape']):
            raise prd.PairingContractError('paired mosaic tile original sizes differ')
        return record


def install_paired_mosaic():
    import paired_rgbir_data as prd
    from tracked_pair_data import install_taps
    cls = prd.DualLabelRGBIRDataset
    if getattr(cls, 'xm_full_r1', False):
        return
    original_getitem = cls.__getitem__

    def getitem(self, index):
        mosaic = None if self.same_modal else active_mosaic(self.base.transforms)
        if mosaic is None:
            return original_getitem(self, index)
        native_get_indexes = mosaic.get_indexes  # bound class method; overridden on the instance only below
        picked = []

        def record():
            indexes = native_get_indexes()
            picked.append(list(indexes))
            return indexes

        def replay():
            native_get_indexes()  # identical RNG consumption to the RGB pass
            if len(picked) != 1:
                raise prd.PairingContractError('IR replay of a mosaic sample without one recorded RGB layout')
            return list(picked[0])

        weak_record = self.base.get_image_and_label(index)
        source = prd._canonical(weak_record['im_file'])
        before = prd._rng_state()
        mosaic.get_indexes = record
        try:
            weak = dict(self.base.transforms(weak_record))
        finally:
            del mosaic.get_indexes
        after = prd._rng_state()
        strong_path = self._mapping[source]
        try:
            strong_record = self._teacher_record(strong_path)
            if tuple(strong_record['ori_shape']) != tuple(weak.get('ori_shape', ())):
                raise prd.PairingContractError('paired original image sizes differ')
            prd._restore_rng(before)
            rgb_dataset, mosaic.dataset, mosaic.get_indexes = mosaic.dataset, IRTiles(self), replay
            try:
                strong = dict(self.base.transforms(strong_record))
            finally:
                mosaic.dataset = rgb_dataset
                del mosaic.get_indexes
        finally:
            prd._restore_rng(after)
        if weak['img'].shape != strong['img'].shape:
            raise prd.PairingContractError('paired image shapes differ after replay')
        OBSERVED['mosaic_samples'] += 1
        output = dict(weak)
        output['strong_img'] = strong['img'].clone()
        output['strong_cls'] = strong['cls'].clone()
        output['strong_bboxes'] = strong['bboxes'].clone()
        output['strong_batch_idx'] = strong['batch_idx'].clone()
        output['pair_info'] = {'weak_source': source, 'strong_source': strong_path,
                               'independent_teacher_gt': not self.same_modal,
                               'mosaic_tile_indexes': list(picked[0]) if picked else []}
        return output

    def close_mosaic(self, hyp):
        self.base.close_mosaic(hyp)
        prd.validate_frozen_geometry(self.base.transforms)
        if hasattr(self, 'geometry_recorder'):
            install_taps(self.base.transforms, self.geometry_recorder)
        OBSERVED['close_mosaic_calls'] += 1

    cls.__getitem__ = getitem
    cls.close_mosaic = close_mosaic
    cls.xm_full_r1 = True


def enable():
    """Called once by scene_runtime.build before any trainer or dataset exists."""
    aug = RECIPE['augmentation']
    if any(float(aug[k]) for k in ('hsv_h', 'hsv_s', 'hsv_v')) or float(aug['mosaic']):
        allow_full_augmentation()
        install_paired_mosaic()

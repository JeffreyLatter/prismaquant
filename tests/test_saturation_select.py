

def test_bisect_mode_is_sublinear_in_measurements():
    # QC on review-batch: the dense default is free on precomputed grids
    # (the only production caller) but live-measurement callers need the
    # documented O(log n) contract back — scan="bisect"/"auto".
    from prismaquant.saturation_select import find_saturation_bpp

    calls = []

    def measure(bpp):
        calls.append(bpp)
        return (0.5 if bpp < 6.0 else 0.1, 0.01)

    grid = [4.0 + 0.25 * i for i in range(17)]  # 17 points
    out = find_saturation_bpp(grid, measure, z=2.0, scan="bisect")
    assert out["bpp"] == 6.0
    assert len(calls) <= 7, f"bisect must be O(log n), measured {len(calls)}"

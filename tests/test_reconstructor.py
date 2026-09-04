from reconstruction._internal.reconstructor import Reconstructor


def test_reconstruct_with_3d_points():
    visual_data = [
        {"point_3d": [1.0, 2.0, 3.0]},
        {"point_3d": [4.0, 5.0, 6.0]},
    ]

    reconstructor = Reconstructor(visual_data, None)
    result = reconstructor.reconstruct()

    assert result["point_cloud"] == [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]
    assert result["mesh"] is None
    assert result["metadata"]["num_points"] == 2


def test_reconstruct_with_no_data():
    reconstructor = Reconstructor(None, None)
    result = reconstructor.reconstruct()

    assert result["point_cloud"] == []
    assert result["mesh"] is None
    assert result["metadata"]["num_points"] == 0


def test_reconstruct_with_triangulation():
    proj1 = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    proj2 = [
        [1.0, 0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]

    visual_data = [
        {"point": [0.0, 0.0], "projection_matrix": proj1},
        {"point": [-0.5, 0.0], "projection_matrix": proj2},
    ]

    reconstructor = Reconstructor(visual_data, None)
    result = reconstructor.reconstruct()

    assert len(result["point_cloud"]) == 1
    assert result["metadata"]["num_points"] == 1


def test_reconstruct_with_poses():
    pose1 = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    pose2 = [
        [1.0, 0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]

    visual_data = [
        {"point": [0.0, 0.0], "pose": pose1},
        {"point": [-0.5, 0.0], "pose": pose2},
    ]

    reconstructor = Reconstructor(visual_data, None)
    result = reconstructor.reconstruct()

    assert len(result["point_cloud"]) == 1
    assert result["metadata"]["num_points"] == 1

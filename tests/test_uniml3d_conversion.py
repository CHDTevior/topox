import numpy as np
import unittest
import tempfile
from pathlib import Path
from scipy.spatial.transform import Rotation

from scripts._build_uniml3d_ktjd17 import Reject, bfs_order, make_skeleton, encode, select_heading_carrier
from scripts._analyze_uniml3d_ktjd17 import tree_signature
from scripts._build_uniml3d_ktjd17 import source_ready
from scripts._download_uniml3d import rate_limit_response


def fixture():
    # Deliberately scrambled parent ordering and non-identity root/child rest rotations.
    names=np.array(["tip","right","root","left"])
    parents=np.array([1,2,-1,2])
    pos=np.array([[0.,.7,0.],[-1.,0.,0.],[4.,2.,-3.],[1.,0.,0.]])
    R=Rotation.from_euler("xyz",[[.2,0,.1],[0,.3,0],[0,.7,0],[.1,0,0]]).as_quat()[:,[3,0,1,2]]
    anim=np.tile(R,(40,1,1)); ap=np.tile(pos,(40,1,1))
    ap[:,2,2]+=np.linspace(0,1,40)
    source=dict(names=names,parents=parents,rest_local_pos=pos,rest_local_rot=R,
                anim_local_pos=ap,anim_local_rot=anim,fps=np.array(30),action_name=np.array("walk"))
    face={"r_hip":{"raw":"right"},"l_hip":{"raw":"left"}}
    return source,face


def test_scrambled_tree_real_rest_and_roundtrip():
    src,face=fixture()
    sk,C,o,perm=make_skeleton(src,face,{n:n for n in src["names"]})
    assert list(sk["joint_names"])==["root","right","left","tip"]
    assert np.array_equal(sk["parents"],[-1,0,0,1])
    assert np.allclose(sk["R_rest_local"][0],sk["R_rest_global"][0])
    assert np.array_equal(sk["offset_parent_local"][0],np.zeros(3))
    m,hv,oxz,qa=encode(src,sk,C,o)
    assert np.max(np.abs(m[...,3:9]-[1,0,0,0,1,0]))<1e-6
    assert hv.all() and np.allclose(m[:,0,15:17],[1,0])
    assert np.array_equal(m[:,1:,13:17],np.zeros_like(m[:,1:,13:17]))
    assert qa["float32_fk_direct_max_norm"]<1e-6
    assert qa["roundtrip_max_scaled"]<1e-6


def test_reject_stretching_and_multiple_roots():
    """两道数据清理闸门的正负例，外加"阈值确实从参数传进来并改变结果"。

    2026-09-20 起旧的 `animated_nonroot_translation`（判据是"最大单关节偏移漂移/骨架尺度 > 1e-3"）
    被换成两道判在**刚化后真实位置误差**上的闸（单位=平均骨长）：
      rigidify_error       每帧关节均值取时间最大 > --rigidify-max（默认 .5）
      rigidify_joint_error 单关节单帧最差      > --rigidify-joint-cap（默认 2.）
    这条测试此前断言的是已被删除的 reason 字符串，等于断言一个永不出现的东西（复审 2026-09-20 点名）。

    fixture 的算术：非根骨长 |right|=1 |left|=1 |tip|=.7 -> 平均 .9；源序 index 0 是叶子 "tip"，
    给它的局部偏移加 d 只影响它自己一个关节（旋转保范数），于是
      单关节误差 = d/.9，       4 个关节里只错 1 个 -> 均值 = d/(4*.9)。
    取 d=2.4：单关节 2.667 > 2. 且均值 .667 > .5，两道闸都够得着，可分别验。"""
    tc=unittest.TestCase()
    src,face=fixture();sk,C,o,_=make_skeleton(src,face,{n:n for n in src["names"]})
    encode(src,sk,C,o)                                    # 未扰动：两道闸都不该触发
    bad,_f=fixture();bad["anim_local_pos"][10,0,0]+=2.4
    with tc.assertRaisesRegex(Reject,"rigidify_error"):encode(bad,sk,C,o)
    # 放开均值闸后，单关节闸必须**单独**拦下 —— 否则 cap 形同虚设
    with tc.assertRaisesRegex(Reject,"rigidify_joint_error"):
        encode(bad,sk,C,o,rigidify_max=float("inf"))
    # 两道都放开就该通过：证明上面的拒绝来自阈值本身，不是别的检查顺带拦的
    encode(bad,sk,C,o,rigidify_max=float("inf"),rigidify_joint_cap=float("inf"))
    # 反向：阈值收紧能拦下本来合格的（-1. 让恒为 0 的误差也超标）
    with tc.assertRaisesRegex(Reject,"rigidify_error"):encode(src,sk,C,o,rigidify_max=-1.)
    with tc.assertRaisesRegex(Reject,"rigidify_joint_error"):
        encode(src,sk,C,o,rigidify_max=float("inf"),rigidify_joint_cap=-1.)
    with tc.assertRaisesRegex(Reject,"multiple_or_missing_roots"):bfs_order(np.array([-1,-1,0]))


def test_topology_census_ignores_sibling_order():
    assert tree_signature([-1,0,0,1])==tree_signature([-1,0,0,2])
    assert tree_signature([-1,0,0,1])!=tree_signature([-1,0,1,2])


def test_stationary_world_root_uses_body_carrier():
    src,face=fixture()
    src["names"]=np.append(src["names"],"world")
    src["parents"]=np.array([1,2,4,2,-1])
    src["rest_local_pos"]=np.concatenate([src["rest_local_pos"],[[4.,2.,-3.]]])
    src["rest_local_pos"][2]=0.
    src["rest_local_rot"]=np.concatenate([src["rest_local_rot"],[[1.,0.,0.,0.]]])
    src["anim_local_pos"]=np.tile(src["rest_local_pos"],(40,1,1))
    src["anim_local_rot"]=np.tile(src["rest_local_rot"],(40,1,1))
    src["anim_local_rot"][:,2]=Rotation.from_euler("y",np.linspace(.7,1.2,40)[:,None]).as_quat()[:,[3,0,1,2]]
    sk,C,o,_=make_skeleton(src,face,{n:n for n in src["names"]})
    carrier,why=select_heading_carrier(sk,[src],C)
    assert str(sk["joint_names"][carrier])=="root" and carrier!=0
    m,hv,_,_=encode(src,sk,C,o,carrier)
    angle=np.unwrap(np.arctan2(m[:,0,16],m[:,0,15]))
    assert angle.max()-angle.min()>.49 and hv.all()


def test_wrapped_rate_limit_and_source_size():
    from requests import Response
    from huggingface_hub.errors import HfHubHTTPError,LocalEntryNotFoundError
    response=Response();response.status_code=429;response.headers["Retry-After"]="12"
    inner=HfHubHTTPError("rate limited",response=response)
    outer=LocalEntryNotFoundError("metadata lookup failed");outer.__cause__=inner
    assert rate_limit_response(outer) is response
    assert rate_limit_response(ValueError("unrelated")) is None
    with tempfile.TemporaryDirectory() as d:
        raw=Path(d);clip={"source_npz":"file.npz","source_bytes":3}
        assert not source_ready(raw,clip)
        (raw/"file.npz").write_bytes(b"ab")
        assert not source_ready(raw,clip)
        (raw/"file.npz").write_bytes(b"abc")
        assert source_ready(raw,clip)


if __name__ == "__main__":
    suite=unittest.TestSuite(unittest.FunctionTestCase(fn) for fn in
        (test_scrambled_tree_real_rest_and_roundtrip,test_reject_stretching_and_multiple_roots,
         test_topology_census_ignores_sibling_order,test_stationary_world_root_uses_body_carrier,
         test_wrapped_rate_limit_and_source_size))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())

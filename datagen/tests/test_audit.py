import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from pxr import Sdf, Usd, UsdGeom, Vt

from strata_datagen.scene import audit, compose, export
from strata_datagen.scene.spec import ShotSpec
from strata_datagen.usdutil import ints, tag_instances, vec3f
from tests.test_determinism import RENDER, hero_files, library

RANGES: list[dict[str, object]] = [
    {"name": "sky", "first_id": 0, "count": 1},
    {"name": "terrain", "first_id": 1, "count": 1},
    {"name": "hero_skin", "first_id": 2, "count": 1},
    {"name": "hero_hair", "first_id": 4, "count": 1},
    {"name": "grass", "first_id": 1000, "count": 3},
]


def _mesh(stage: Usd.Stage, path: str) -> Usd.Prim:
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(vec3f(np.eye(3)))
    mesh.CreateFaceVertexCountsAttr(ints(np.array([3])))
    mesh.CreateFaceVertexIndicesAttr(ints(np.arange(3)))
    return mesh.GetPrim()


def _scene() -> Usd.Stage:
    """Every kind of rendered surface a shot authors, each tagged the way the generators tag it."""
    part = Usd.Stage.CreateInMemory()
    part.SetDefaultPrim(UsdGeom.Xform.Define(part, "/part").GetPrim())
    _mesh(part, "/part/body")
    stage = Usd.Stage.CreateInMemory()
    sky = UsdGeom.Sphere.Define(stage, "/World/sky").GetPrim()
    tag_instances(sky, np.array([0]), "sky")
    tag_instances(_mesh(stage, "/World/terrain"), np.array([1]), "terrain")
    hero = UsdGeom.Xform.Define(stage, "/World/hero").GetPrim()
    hero.GetReferences().AddReference(part.GetRootLayer().identifier)
    hero.SetInstanceable(instanceable=True)
    tag_instances(hero, np.array([2]), "hero_skin")
    hair = UsdGeom.BasisCurves.Define(stage, "/World/hair")
    hair.CreateCurveVertexCountsAttr(ints(np.array([2])))
    hair.CreatePointsAttr(vec3f(np.zeros((2, 3))))
    tag_instances(hair.GetPrim(), np.array([4]), "hero_hair")
    instancer = UsdGeom.PointInstancer.Define(stage, "/World/grass")
    prototype = _mesh(stage, "/World/grass/prototypes/p0")
    instancer.CreatePrototypesRel().SetTargets([prototype.GetPath()])
    instancer.CreatePositionsAttr(vec3f(np.zeros((3, 3))))
    instancer.CreateProtoIndicesAttr(ints(np.zeros(3, np.int64)))
    tag_instances(instancer.GetPrim(), np.arange(1000, 1003), "grass")
    return stage


class TagAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.stage = _scene()

    def errors(self) -> list[str]:
        return audit.tag_errors(self.stage, RANGES)

    def test_a_correctly_tagged_scene_passes_through_instances_and_instancers(self) -> None:
        self.assertEqual(self.errors(), [])
        proxies = Usd.PrimRange.Stage(self.stage, Usd.TraverseInstanceProxies())
        self.assertIn("/World/hero/body", [str(p.GetPath()) for p in proxies if p.IsInstanceProxy()])

    def test_a_surface_missing_a_role_primvar_fails(self) -> None:
        for path in ("/World/terrain", "/World/sky", "/World/hair"):
            with self.subTest(path=path):
                self.stage = _scene()
                self.stage.GetPrimAtPath(path).RemoveProperty("primvars:hero")
                self.assertEqual(self.errors(), [f"{path}: no authored hero"])

    def test_geometry_inside_an_instanceable_reference_needs_its_inherited_roles(self) -> None:
        self.stage.GetPrimAtPath("/World/hero").RemoveProperty("primvars:fur")
        self.assertEqual(self.errors(), ["/World/hero/body: no authored fur"])

    def test_an_untagged_surface_fails_instead_of_rendering_as_sky(self) -> None:
        _mesh(self.stage, "/World/rock")
        self.assertEqual(
            self.errors(),
            ["/World/rock: no authored instance_id, instance_check, instance_check2, foreground, hero, fur"],
        )

    def test_role_values_must_match_the_membership_of_the_id_range(self) -> None:
        primvar = UsdGeom.PrimvarsAPI(self.stage.GetPrimAtPath("/World/hero")).GetPrimvar("hero")
        primvar.Set(0.0)
        self.assertEqual(self.errors(), ["/World/hero/body: hero differ from the roles of their id range"])
        primvar.Set(1.0)
        UsdGeom.PrimvarsAPI(self.stage.GetPrimAtPath("/World/terrain")).GetPrimvar("instance_id").Set(3)
        self.assertEqual(self.errors(), ["/World/terrain: instance ids [3] outside the manifest ranges"])

    def test_an_instancer_missing_a_role_array_fails_for_its_prototypes(self) -> None:
        self.stage.GetPrimAtPath("/World/grass").RemoveProperty("primvars:foreground")
        self.assertEqual(self.errors(), ["/World/grass: no authored foreground"])

    def test_instancer_arrays_need_one_value_per_instance(self) -> None:
        api = UsdGeom.PrimvarsAPI(self.stage.GetPrimAtPath("/World/grass"))
        api.GetPrimvar("fur").Set([0.0, 0.0])
        self.assertEqual(self.errors(), ["/World/grass: role primvars do not have one value per instance"])
        api.GetPrimvar("fur").Set([0.0, 1.0, 0.0])
        self.assertEqual(self.errors(), ["/World/grass: fur differ from the roles of their id range"])
        api.GetPrimvar("fur").Set([0.0, 0.0, 0.0])
        self.stage.GetPrimAtPath("/World/grass").GetAttribute("protoIndices").Set(ints(np.zeros(4, np.int64)))
        self.assertEqual(self.errors(), ["/World/grass: 3 instance ids for 4 instances"])

    def test_prototype_geometry_must_not_override_its_instancer_tags(self) -> None:
        prototype = self.stage.GetPrimAtPath("/World/grass/prototypes/p0")
        UsdGeom.PrimvarsAPI(prototype).CreatePrimvar(
            "hero", Sdf.ValueTypeNames.Float, UsdGeom.Tokens.constant
        ).Set(0.0)
        self.assertEqual(
            self.errors(), ["/World/grass/prototypes/p0: prototype geometry overrides its instancer's hero"]
        )

    def test_guides_are_not_rendered_and_need_no_tags(self) -> None:
        UsdGeom.Imageable(_mesh(self.stage, "/World/guide")).CreatePurposeAttr(UsdGeom.Tokens.guide)
        self.assertEqual(self.errors(), [])


def _primvar(stage: Usd.Stage, path: str, name: str) -> UsdGeom.Primvar:
    return UsdGeom.PrimvarsAPI(stage.GetPrimAtPath(path)).GetPrimvar(name)


def _sample(primvar: UsdGeom.Primvar) -> None:
    primvar.Set(primvar.Get(), 1.0)


def _connect(primvar: UsdGeom.Primvar) -> None:
    primvar.GetAttr().AddConnection("/World/terrain.points")


def _index(primvar: UsdGeom.Primvar) -> None:
    primvar.GetAttr().GetPrim().CreateAttribute(
        f"primvars:{primvar.GetPrimvarName()}:indices", Sdf.ValueTypeNames.IntArray
    ).Set(Vt.IntArray([0]))


def _element_size(primvar: UsdGeom.Primvar) -> None:
    primvar.SetElementSize(2)


def _interpolate(primvar: UsdGeom.Primvar) -> None:
    primvar.SetInterpolation(UsdGeom.Tokens.uniform)


FAULTS = {
    "has time samples": _sample,
    "has attribute connections": _connect,
    "is indexed": _index,
    "has an element size other than 1": _element_size,
    "has uniform interpolation": _interpolate,
}
SURFACES = {
    "/World/terrain": "/World/terrain",
    "/World/sky": "/World/sky",
    "/World/hair": "/World/hair",
    "/World/hero": "/World/hero/body",
}


class StaticTagTest(unittest.TestCase):
    def test_every_semantic_primvar_fault_is_rejected_on_every_kind_of_surface(self) -> None:
        for owner, rendered in SURFACES.items():
            for name in audit.TAGS:
                for why, fault in FAULTS.items():
                    with self.subTest(owner=owner, name=name, fault=why):
                        stage = _scene()
                        fault(_primvar(stage, owner, name))
                        inherits = owner != "/World/hero" or why != "has uniform interpolation"
                        expected = (
                            f"{rendered}: {name} {why}" if inherits else f"{rendered}: no authored {name}"
                        )
                        self.assertIn(expected, audit.tag_errors(stage, RANGES))

    def test_time_samples_that_disagree_with_the_default_are_rejected(self) -> None:
        stage = _scene()
        primvar = _primvar(stage, "/World/hero", "hero")
        primvar.Set(0.0, 1.0)
        self.assertEqual(primvar.Get(), 1.0)
        self.assertIn("/World/hero/body: hero has time samples", audit.tag_errors(stage, RANGES))

    def test_instancer_arrays_reject_the_same_faults(self) -> None:
        for name in audit.TAGS:
            for why, fault in FAULTS.items():
                if why == "has uniform interpolation":
                    continue
                with self.subTest(name=name, fault=why):
                    stage = _scene()
                    fault(_primvar(stage, "/World/grass", name))
                    self.assertIn(f"/World/grass: {name} {why}", audit.tag_errors(stage, RANGES))

    def test_instancer_arrays_reject_interpolations_other_than_vertex_and_constant(self) -> None:
        stage = _scene()
        _interpolate(_primvar(stage, "/World/grass", "hero"))
        self.assertIn("/World/grass: hero has uniform interpolation", audit.tag_errors(stage, RANGES))

    def test_prototype_geometry_with_time_sampled_tags_is_rejected(self) -> None:
        stage = _scene()
        path = "/World/grass/prototypes/p0"
        primvar = UsdGeom.PrimvarsAPI(stage.GetPrimAtPath(path)).CreatePrimvar(
            "hero", Sdf.ValueTypeNames.Float, UsdGeom.Tokens.constant
        )
        primvar.Set(1.0, 1.0)
        self.assertEqual(
            audit.tag_errors(stage, RANGES), [f"{path}: prototype geometry overrides its instancer's hero"]
        )

    def test_value_clips_that_override_a_referenced_tag_are_rejected_and_unrelated_clips_pass(self) -> None:
        with TemporaryDirectory() as tmp:
            for supplied, expected in (("primvars:hero", True), ("points", False)):
                with self.subTest(supplied=supplied):
                    case = Path(tmp, supplied.replace(":", "_"))
                    case.mkdir()
                    asset = Usd.Stage.CreateNew(f"{case}/part.usda")
                    tag_instances(_mesh(asset, "/part"), np.array([1]), "terrain")
                    asset.GetRootLayer().Save()
                    clip = Usd.Stage.CreateNew(f"{case}/clip.usda")
                    mesh = clip.DefinePrim("/clip", "Mesh")
                    if supplied == "points":
                        mesh.CreateAttribute(supplied, Sdf.ValueTypeNames.Point3fArray).Set(
                            vec3f(np.eye(3)), 1.0
                        )
                    else:
                        mesh.CreateAttribute(supplied, Sdf.ValueTypeNames.Float).Set(1.0, 1.0)
                    clip.GetRootLayer().Save()
                    scene = _scene()
                    prim = scene.GetPrimAtPath("/World/terrain")
                    for name in [n for n in prim.GetPropertyNames() if n.startswith("primvars:")]:
                        prim.RemoveProperty(name)
                    prim.GetReferences().AddReference(f"{case}/part.usda", "/part")
                    api = Usd.ClipsAPI(prim)
                    api.SetClipAssetPaths([Sdf.AssetPath(f"{case}/clip.usda")])
                    api.SetClipManifestAssetPath(Sdf.AssetPath(f"{case}/clip.usda"))
                    api.SetClipPrimPath("/clip")
                    api.SetClipActive([(0.0, 0)])
                    api.SetClipTimes([(0.0, 0.0)])
                    errors = audit.tag_errors(scene, RANGES)
                    self.assertEqual("/World/terrain: foreground is under value clips" in errors, False)
                    self.assertEqual("/World/terrain: hero is under value clips" in errors, expected)
                    self.assertEqual(any("has time samples" in error for error in errors), expected)

    def test_non_binary_roles_are_rejected(self) -> None:
        stage = _scene()
        _primvar(stage, "/World/terrain", "foreground").Set(0.5)
        self.assertEqual(
            audit.tag_errors(stage, RANGES),
            ["/World/terrain: foreground differ from the roles of their id range"],
        )


class BuildAuditTest(unittest.TestCase):
    def test_a_built_scene_with_a_missing_role_primvar_gets_no_manifest(self) -> None:
        tmp = self.enterContext(TemporaryDirectory())
        root = Path(tmp)
        lib = library()
        hero_files(root, lib)
        shot = compose.plan(ShotSpec(seed=7), lib, root)
        path = compose.write(shot, lib, root, RENDER)
        stage = Usd.Stage.Open(str(path))
        stage.GetPrimAtPath("/World/hero/skin").RemoveProperty("primvars:hero")
        stage.GetRootLayer().Save()
        with self.assertRaisesRegex(ValueError, "/World/hero/skin: no authored hero"):
            export.write(shot, lib, RENDER)
        self.assertFalse((shot.directory / "manifest.json").exists())
        self.assertFalse((shot.directory / "camera.json").exists())

    def test_a_built_scene_with_a_time_sampled_hero_role_gets_no_manifest(self) -> None:
        tmp = self.enterContext(TemporaryDirectory())
        root = Path(tmp)
        lib = library()
        hero_files(root, lib)
        shot = compose.plan(ShotSpec(seed=7), lib, root)
        path = compose.write(shot, lib, root, RENDER)
        stage = Usd.Stage.Open(str(path))
        _primvar(stage, "/World/hero/skin", "hero").Set(0.0, 1.0)
        stage.GetRootLayer().Save()
        with self.assertRaisesRegex(ValueError, "/World/hero/skin: hero has time samples"):
            export.write(shot, lib, RENDER)
        self.assertFalse((shot.directory / "manifest.json").exists())
        self.assertFalse((shot.directory / "camera.json").exists())

import base64
import hashlib
import io
import importlib.util
import json
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app import app
from model import Inspector, MAX_BYTES, decode_image, heatmap_png, patch_distances

ROOT = Path(__file__).resolve().parents[1]

class InspectionTests(unittest.TestCase):
    def test_api_imports_when_vercel_omits_public_directory(self):
        spec = importlib.util.spec_from_file_location("vercel_app_check", ROOT / "app.py")
        module = importlib.util.module_from_spec(spec)
        with patch.object(Path, "is_dir", return_value=False):
            spec.loader.exec_module(module)
        client = TestClient(module.app)
        self.assertEqual(client.get("/api/health").status_code, 200)
        self.assertEqual(client.get("/").status_code, 404)

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.inspector = Inspector()
        cls.samples = json.loads((ROOT / "public/samples.json").read_text())

    def test_memory_distance_matches_known_vectors(self):
        features = np.array([[[[0.,3.]],[[0.,4.]]]], np.float32)
        distance = patch_distances(features, np.array([[0.,0.]], np.float32), np.ones(2, np.float32))
        np.testing.assert_allclose(distance, [[0.,5.]])

    def test_low_field_not_exaggerated_by_heatmap(self):
        content = base64.b64decode(heatmap_png(np.zeros((20,20), np.float32), 10))
        rgba = np.asarray(Image.open(io.BytesIO(content)))
        self.assertEqual(rgba.shape, (320,320,4)); self.assertEqual(int(rgba[:,:,3].max()), 0)

    def test_artifacts_match_recorded_hashes(self):
        for filename, digest in self.inspector.metadata["files_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT/"artifacts"/filename).read_bytes()).hexdigest(), digest)

    def test_split_is_disjoint_and_complete(self):
        evaluation = json.loads((ROOT/"artifacts/evaluation.json").read_text())
        fit, cal = set(evaluation["fit_paths"]), set(evaluation["calibration_paths"])
        self.assertFalse(fit & cal); self.assertEqual((len(fit),len(cal)),(167,42))
        for method in evaluation["methods"].values():
            self.assertEqual(len(method["records"]), 83)
            self.assertTrue(all('/test/' in row["path"] for row in method["records"]))

    def test_health_exposes_version_and_no_image_storage(self):
        response = self.client.get('/api/health'); self.assertEqual(response.status_code,200)
        self.assertFalse(response.json()["uploads_stored"])
        self.assertEqual(response.json()["model"],self.inspector.metadata["model_version"])

    def test_real_inference_matches_recorded_sample(self):
        sample = self.samples[1]; data = (ROOT/'public'/sample['file']).read_bytes()
        response = self.client.post('/api/inspect',content=data,headers={'Content-Type':'image/jpeg'})
        self.assertEqual(response.status_code,200)
        result = response.json()
        self.assertAlmostEqual(result['score'],sample['recorded_result']['score'],delta=.01)
        self.assertEqual(result['decision'],sample['recorded_result']['decision'])
        self.assertTrue(result['heatmap'].startswith('data:image/png;base64,'))
        self.assertEqual(response.headers['cache-control'],'no-store')

    def test_normal_and_defect_have_distinct_model_signals(self):
        results = [self.inspector.inspect((ROOT/'public'/s['file']).read_bytes()) for s in self.samples[:2]]
        self.assertLess(results[0]['score'],results[1]['score'])

    def test_invalid_image_is_client_error(self):
        self.assertEqual(self.client.post('/api/inspect',content=b'not an image').status_code,422)

    def test_empty_image_is_client_error(self):
        self.assertEqual(self.client.post('/api/inspect',content=b'').status_code,422)

    def test_oversize_request_is_rejected(self):
        self.assertEqual(self.client.post('/api/inspect',content=b'x'*(MAX_BYTES+1)).status_code,413)

    def test_small_image_is_rejected(self):
        stream = io.BytesIO(); Image.new('RGB',(8,8)).save(stream,format='PNG')
        with self.assertRaisesRegex(ValueError,'too small'): decode_image(stream.getvalue())

    def test_gif_is_rejected_even_if_decodable(self):
        stream = io.BytesIO(); Image.new('RGB',(40,40)).save(stream,format='GIF')
        with self.assertRaisesRegex(ValueError,'JPEG, PNG or WebP'): decode_image(stream.getvalue())

    def test_frontend_and_samples_are_served(self):
        self.assertEqual(self.client.get('/').status_code,200)
        self.assertEqual(self.client.get('/samples.json').status_code,200)

if __name__ == '__main__': unittest.main()

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'Benchmarks/Tools'))
from pretrained_llama import verify_checkpoint,MANIFEST,PROMPTS


class PretrainedLlamaTests(unittest.TestCase):
    def test_verified_artifact_rejects_tampering_and_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);p=directory/'model.safetensors';p.write_bytes(b'original')
            manifest=dict(artifacts=[dict(path=p.name,bytes=8,sha256=hashlib.sha256(b'original').hexdigest())])
            verify_checkpoint(directory,manifest)
            p.write_bytes(b'changed!')
            with self.assertRaises(ValueError):verify_checkpoint(directory,manifest)
            p.unlink()
            with self.assertRaises(ValueError):verify_checkpoint(directory,manifest)

    def test_artifact_path_cannot_escape_checkpoint(self):
        for path in ['../model.safetensors','/tmp/model.safetensors']:
            with self.assertRaises(ValueError):verify_checkpoint(Path('/tmp'),dict(artifacts=[dict(path=path)]))

    def test_pinned_checkpoint_and_prompt_contract(self):
        manifest=json.loads(MANIFEST.read_text())
        self.assertEqual(len(manifest['revision']),40)
        self.assertEqual(manifest['format'],'safetensors-bfloat16')
        names=[x['path'] for x in manifest['artifacts']]
        self.assertEqual(len(names),len(set(names)))
        self.assertTrue({'model.safetensors','config.json','tokenizer.json','tokenizer_config.json'}<=set(names))
        for artifact in manifest['artifacts']:
            self.assertEqual(len(artifact['sha256']),64);self.assertGreater(artifact['bytes'],0)
        cases=json.loads(PROMPTS.read_text())['cases']
        self.assertEqual(len(cases),len({c['id'] for c in cases}))
        self.assertTrue(all(c['text'] for c in cases))

    def test_unlisted_checkpoint_inputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            payload = b'original'
            (directory / 'model.safetensors').write_bytes(payload)
            manifest = dict(artifacts=[dict(path='model.safetensors', bytes=len(payload),
                                           sha256=hashlib.sha256(payload).hexdigest())])
            cache = directory / '.cache'
            cache.mkdir()
            (cache / 'download-metadata').write_text('local download bookkeeping')
            verify_checkpoint(directory, manifest)
            for name in ['model-extra.safetensors', 'generation_config.json']:
                extra = directory / name
                extra.write_text('unverified loader input')
                with self.assertRaisesRegex(ValueError, 'inventory'):
                    verify_checkpoint(directory, manifest)
                extra.unlink()

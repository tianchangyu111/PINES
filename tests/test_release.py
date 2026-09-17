"""Source-release checks requiring only the Python standard library."""
import ast
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
EXPECTED = {'hybrid_pipeline', 'transfer_learning', 'gas_transfer',
            'feature_classifier_comparison', 'localization_comparison',
            'resnet_pipeline', 'resnet_inference', 'lightgbm_inference',
            'concentration_classifier'}
LEGACY = {'biosensor_hybrid_v6_pycharm', 'transfer_learning_only',
          'gas_hybrid_transfer', 'gas_29feat_classifiers',
          'gas_resnet29_stage2a_compare', 'resnet29_full_parity',
          'resnet29_unknown_infer', 'lightgbm_unknown_infer'}


class ReleaseChecks(unittest.TestCase):
    def test_nine_research_scripts(self):
        self.assertEqual({p.stem for p in SCRIPTS.glob('*.py')}, EXPECTED)

    def test_syntax_and_local_imports(self):
        for path in SCRIPTS.glob('*.py'):
            source = path.read_text(encoding='utf-8')
            tree = ast.parse(source, filename=path.name)
            compile(tree, path.name, 'exec')
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn(node.module, LEGACY)
                    if node.module in EXPECTED:
                        self.assertTrue((SCRIPTS / (node.module + '.py')).is_file())

    def test_no_cjk_release_text(self):
        cjk = re.compile(r'[\u3400-\u9fff]')
        for path in ROOT.rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc':
                continue
            self.assertTrue(str(path.relative_to(ROOT)).isascii(), str(path))
            if path.suffix != '.pt':
                self.assertIsNone(cjk.search(path.read_text(encoding='utf-8')), str(path))

    def test_deployment_python_syntax(self):
        path = ROOT / 'deployment' / 'raspberry_pi' / 'cam_server.py'
        source = path.read_text(encoding='utf-8')
        compile(source, str(path), 'exec')

    def test_no_personal_windows_paths(self):
        for path in SCRIPTS.glob('*.py'):
            self.assertIsNone(re.search(r'(?<![A-Za-z])[A-Za-z]:[\\/]', path.read_text(encoding='utf-8')), path.name)


if __name__ == '__main__':
    unittest.main()

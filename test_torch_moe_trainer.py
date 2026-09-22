import json
import shutil
import subprocess
import unittest
import torch
from pathlib import Path
import torch_moe_trainer as trainer

ROOT=Path(__file__).parent
PYTHON=ROOT/'.venv-torch/bin/python'
OUT=ROOT/'.tmp-torch-moe-test'

class TorchMoETrainerTest(unittest.TestCase):
    def test_unsupported_policy_margin_penalizes_a_legal_zero_probability_move_above_supported_moves(self):
        logits=torch.tensor([[0.0, 0.5, 2.0]])
        probabilities=torch.tensor([[1.0, 0.0, 0.0]])
        legal=torch.tensor([[True, True, False]])
        loss=trainer.unsupported_policy_margin_loss(logits,probabilities,legal,margin=1.0)
        self.assertAlmostEqual(loss.item(),1.5)

    def test_records_unsupported_policy_margin_in_browser_snapshot(self):
        output=ROOT/'.tmp-unsupported-margin-output'
        shutil.rmtree(output,ignore_errors=True)
        subprocess.run([str(PYTHON),'torch_moe_trainer.py','--epochs','1','--batch-size','256','--unsupported-margin','0.1','--output',str(output),'--snapshot-debounce-seconds','0'],cwd=ROOT,check=True,timeout=180)
        snapshot=json.loads(next(output.glob('palace-9-round-*.json')).read_text())
        self.assertEqual(snapshot['format'],'palace-9-moe/v1')
        self.assertEqual(snapshot['annotations']['unsupportedPolicyMargin'],0.1)

    def test_exports_16_node_browser_model(self):
        shutil.rmtree(OUT,ignore_errors=True)
        result=subprocess.run([str(PYTHON),'torch_moe_trainer.py','--epochs','1','--batch-size','256','--hidden-nodes','16','--expert-nodes','16','--output',str(OUT)],cwd=ROOT,text=True,capture_output=True,check=True,timeout=180)
        report=json.loads(result.stdout.strip().splitlines()[-1])
        self.assertTrue(report['device'].startswith('cuda:0'))
        files=sorted(OUT.glob('palace-9-round-*.json'))
        self.assertTrue(files)
        self.assertEqual(files[-1].name,'palace-9-round-000001.json')
        model=json.loads(files[-1].read_text())
        self.assertEqual(model['format'],'palace-9-moe/v1')
        self.assertEqual(model['architecture'],'board-state-moe')
        self.assertEqual(model['trunk']['widths'],[9,16,16])
        self.assertTrue(all(expert['widths']==[16,16,9] for expert in model['experts']))
        subprocess.run(['node','-e',"import('./training.mjs').then(({importModel})=>importModel(require(process.argv[1])))",str(files[-1])],cwd=ROOT,check=True,timeout=60)

    def test_debounces_improved_snapshots_and_records_generation_timing(self):
        shutil.rmtree(OUT,ignore_errors=True)
        subprocess.run([str(PYTHON),'torch_moe_trainer.py','--epochs','4','--batch-size','256','--output',str(OUT),'--snapshot-debounce-seconds','3600'],cwd=ROOT,text=True,capture_output=True,check=True,timeout=180)
        files=sorted(OUT.glob('palace-9-round-*.json'))
        self.assertEqual(len(files),1)
        annotations=json.loads(files[0].read_text())['annotations']
        self.assertEqual(annotations['generatedRound'],1)
        self.assertEqual(annotations['rounds'],1)
        self.assertEqual(annotations['elapsedSecondsSinceFirstSnapshot'],0.0)
        self.assertEqual(annotations['averageSecondsPerRoundSinceFirstSnapshot'],0.0)

    def test_resumes_weights_and_continues_round_numbers(self):
        source=ROOT/'torch-snapshots-16'/'palace-9-round-7426.json'
        self.assertTrue(source.exists(), 'known best checkpoint is required for continuation')
        shutil.rmtree(OUT,ignore_errors=True)
        result=subprocess.run([str(PYTHON),'torch_moe_trainer.py','--epochs','1','--batch-size','256','--resume',str(source),'--output',str(OUT),'--snapshot-debounce-seconds','0'],cwd=ROOT,text=True,capture_output=True,check=True,timeout=180)
        report=json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(report['resumedFrom'],str(source))
        self.assertEqual(report['startingRound'],7426)
        self.assertEqual(report['finalRound'],7427)

    def test_trains_invalid_history_sentinel_as_tenth_output(self):
        output=ROOT/'.tmp-sentinel-output'
        shutil.rmtree(output,ignore_errors=True)
        result=subprocess.run([str(PYTHON),'torch_moe_trainer.py','--epochs','1','--batch-size','256','--sentinel','--output',str(output),'--snapshot-debounce-seconds','0'],cwd=ROOT,text=True,capture_output=True,check=True,timeout=180)
        report=json.loads(result.stdout.strip().splitlines()[-1])
        snapshot=json.loads(next(output.glob('palace-9-round-*.json')).read_text())
        self.assertEqual(snapshot['format'],'palace-9-moe/v3')
        self.assertEqual(snapshot['outputTokens'],list('abcdefghi!'))
        self.assertEqual(snapshot['encoder']['widths'],[31,9])
        self.assertTrue(all(expert['widths']==[16,16,10] for expert in snapshot['experts']))
        self.assertGreater(report['invalidHistoryTotal'],0)

    def test_uses_real_invalid_history_categories_and_preserves_symmetry_targets(self):
        rows=json.loads((ROOT/'data/reachable-policy.json').read_text())['examples']
        invalid=trainer.invalid_history_examples(rows)
        categories={row['category'] for row in invalid}
        self.assertTrue({'repeated-square','non-square-token','post-terminal'}<=categories)
        self.assertTrue(all(len(row['features'])==31 and row['target'][-1]==1 for row in invalid))
        source={'features':[1,0,0,0,1,0,0,0,1]+[0]*20,'target':[.1,.2,.3,.4,0,0,0,0,0]}
        rotated=trainer.transform_example(source,trainer.D4_PERMUTATIONS[1])
        self.assertEqual(rotated['features'][:9],[0,0,1,0,1,0,1,0,0])
        self.assertEqual(rotated['target'][:3],[.3,.2,.1])

    def test_canonicalization_selects_a_deterministic_d4_orientation_and_can_restore_logits(self):
        source={'features':[1,0,0,0,1,0,0,0,1]+[0]*22,'target':[.1,.2,.3,.4,0,0,0,0,0,1]}
        canonical=trainer.canonicalize_example(source)
        self.assertEqual(canonical['transformIndex'], 2)
        self.assertEqual(canonical['target'][5:9],[.4,.3,.2,.1])
        self.assertEqual(trainer.inverse_permute_logits(canonical['target'],canonical['permutation'])[:4],[.1,.2,.3,.4])

    def test_canonicalized_valid_corpus_has_no_policy_collisions(self):
        rows=json.loads((ROOT/'data/reachable-policy.json').read_text())['examples']
        seen={}
        for row in rows:
            canonical=trainer.canonicalize_example({'features':row['features']+[0,0],'target':row['policy']['probabilities']+[0]})
            key=tuple(canonical['features'])
            prior=seen.setdefault(key,tuple(canonical['target']))
            self.assertEqual(prior,tuple(canonical['target']))
        self.assertEqual(len(seen),626)

    def test_exports_qat_provenance(self):
        self.assertTrue(torch.allclose(trainer.fake_quantize_ste(torch.tensor([[1.0,.1],[.1,.01]]),4,'row'),torch.tensor([[1.0,.14285714285714285],[.1,.014285714285714287]]),atol=1e-6))
        output=ROOT/'.tmp-qatig-output'
        shutil.rmtree(output,ignore_errors=True)
        subprocess.run([str(PYTHON),'torch_moe_trainer.py','--epochs','1','--batch-size','256','--sentinel','--qat-format','int4','--output',str(output),'--snapshot-debounce-seconds','0'],cwd=ROOT,text=True,capture_output=True,check=True,timeout=180)
        annotations=json.loads(next(output.glob('palace-9-round-*.json')).read_text())['annotations']
        self.assertEqual(annotations['fakeQuant']['format'],'int4')
        self.assertEqual(annotations['fakeQuant']['method'],'symmetric per-tensor STE round/dequantize')

    def test_fp16_fake_quantization_uses_ieee_half_rounding(self):
        weight=torch.tensor([[1.0,0.1,-0.1]],dtype=torch.float32)
        expected=weight.to(torch.float16).to(torch.float32)
        self.assertTrue(torch.equal(trainer.fake_quantize_ste(weight,'float16'),expected))

    def test_synchronized_precision_formats_include_master_and_all_requested_quantizations(self):
        self.assertEqual(trainer.synchronized_precision_formats(),(None,'float16','int8','int4'))

    def test_exports_int8_qat_provenance(self):
        output=ROOT/'.tmp-qat-int8-output'
        shutil.rmtree(output,ignore_errors=True)
        subprocess.run([str(PYTHON),'torch_moe_trainer.py','--epochs','1','--batch-size','256','--sentinel','--qat-format','int8','--output',str(output),'--snapshot-debounce-seconds','0'],cwd=ROOT,text=True,capture_output=True,check=True,timeout=180)
        annotations=json.loads(next(output.glob('palace-9-round-*.json')).read_text())['annotations']
        self.assertEqual(annotations['fakeQuant']['format'],'int8')
        self.assertEqual(annotations['fakeQuant']['storageBits'],8)
        self.assertEqual(annotations['fakeQuant']['method'],'symmetric per-tensor STE round/dequantize')

    def test_can_resume_weights_without_old_optimizer_state(self):
        source_out=ROOT/'.tmp-adamw-source'
        resumed_out=ROOT/'.tmp-adamw-fresh'
        shutil.rmtree(source_out,ignore_errors=True);shutil.rmtree(resumed_out,ignore_errors=True)
        subprocess.run([str(PYTHON),'torch_moe_trainer.py','--epochs','1','--batch-size','256','--output',str(source_out),'--snapshot-debounce-seconds','0'],cwd=ROOT,text=True,capture_output=True,check=True,timeout=180)
        source=next(source_out.glob('palace-9-round-*.json'))
        result=subprocess.run([str(PYTHON),'torch_moe_trainer.py','--epochs','1','--batch-size','256','--resume',str(source),'--fresh-optimizer','--output',str(resumed_out),'--snapshot-debounce-seconds','0'],cwd=ROOT,text=True,capture_output=True,check=True,timeout=180)
        self.assertFalse(json.loads(result.stdout.strip().splitlines()[-1])['optimizerStateRestored'])

    def test_resumes_saved_adamw_state(self):
        source_out=ROOT/'.tmp-adamw-source'
        resumed_out=ROOT/'.tmp-adamw-resumed'
        shutil.rmtree(source_out,ignore_errors=True);shutil.rmtree(resumed_out,ignore_errors=True)
        subprocess.run([str(PYTHON),'torch_moe_trainer.py','--epochs','1','--batch-size','256','--output',str(source_out),'--snapshot-debounce-seconds','0'],cwd=ROOT,text=True,capture_output=True,check=True,timeout=180)
        source=next(source_out.glob('palace-9-round-*.json'))
        optimizer_state=source.with_suffix('.optimizer.pt')
        self.assertTrue(optimizer_state.exists())
        result=subprocess.run([str(PYTHON),'torch_moe_trainer.py','--epochs','1','--batch-size','256','--resume',str(source),'--output',str(resumed_out),'--snapshot-debounce-seconds','0'],cwd=ROOT,text=True,capture_output=True,check=True,timeout=180)
        report=json.loads(result.stdout.strip().splitlines()[-1])
        self.assertTrue(report['optimizerStateRestored'])

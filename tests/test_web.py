import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from src.core.zones import ZoneAnalytics
from src.utils.helpers import Detection
from src.services.storage import atomic_json
import server.app as web


class SpatialTests(unittest.TestCase):
    def test_overlapping_zones_count_union_once_and_heatmap_is_bounded(self):
        zones = [dict(name='A',capacity=1,polygon=None),dict(name='B',capacity=2,polygon=None)]
        a = ZoneAnalytics(zones)
        ds = [Detection(10,10,20,20,.9),Detection(50,50,60,60,.9)]
        for _ in range(4):
            self.assertEqual(len(a.observe(ds,(100,100))),2)
        results, heat = a.finish([0,1,2,3],2,1)
        self.assertEqual(results[0]['counts'],[2]*4)
        self.assertEqual(len(results[0]['events']),1)
        self.assertEqual(results[1]['events'],[])
        self.assertEqual(sum(map(sum,heat['grid'])),8)
        self.assertEqual(len(heat['grid']),36)
    def test_duplicate_and_bad_zone_settings_rejected(self):
        for zones in ([],[dict(name='A',capacity=0,polygon=None)],[dict(name='A',capacity=1,polygon=None)]*2):
            with self.assertRaises(ValueError): ZoneAnalytics(zones)


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.sessions=self.root/'sessions'/web.OWNER
        self.sessions.mkdir(parents=True)
        self.patches=[patch.object(web,'ROOT',self.root),patch.object(web,'SESSIONS',self.sessions),
                      patch.dict(os.environ,{'CROWD_DEV_AUTH':'0','CROWD_ACCESS_PASSWORD':'test-password-long','CROWD_SESSION_SECRET':'s'*40})]
        for p in self.patches:p.start()
        web.attempts.clear()
        self.client=TestClient(web.app)
        self.headers={'X-Campus-Request':'1'}
    def tearDown(self):
        self.client.close()
        for p in reversed(self.patches):p.stop()
        self.tmp.cleanup()
    def login(self):
        r=self.client.post('/api/login',json={'password':'test-password-long'},headers=self.headers)
        self.assertEqual(r.status_code,200)
        self.assertIn('HttpOnly',r.headers['set-cookie'])
    def test_media_requires_auth_and_mutations_require_header(self):
        self.assertEqual(self.client.get('/api/sessions').status_code,401)
        self.assertEqual(self.client.get('/api/sessions/'+'a'*32+'/media/poster').status_code,401)
        self.assertEqual(self.client.post('/api/login',json={'password':'x'}).status_code,403)
        self.login()
        self.assertEqual(self.client.get('/api/sessions').status_code,200)
        self.assertEqual(self.client.post('/api/logout',headers={**self.headers,'Origin':'https://evil.example'}).status_code,403)
        self.assertEqual(self.client.post('/api/logout',headers=self.headers,content=b'x'*65537).status_code,413)
    def test_bad_upload_is_rejected_without_artifacts(self):
        self.login()
        self.assertEqual(self.client.post('/api/upload',content=b'bad',headers={**self.headers,'X-File-Name':'bad.txt'}).status_code,415)
        self.assertEqual(self.client.post('/api/upload',content=b'bad',headers={**self.headers,'X-File-Name':'bad.mp4'}).status_code,422)
        self.assertEqual(list(self.sessions.iterdir()),[])
    def test_busy_worker_and_immutable_result_and_cancel(self):
        self.login();key='a'*32;path=self.sessions/key;path.mkdir()
        atomic_json(path/'metadata.json',dict(created=1,file_name='clip.mp4',source='source.mp4'))
        self.assertIsNone(self.client.get(f'/api/sessions/{key}').json()['result'])
        atomic_json(path/'status.json',dict(state='running'))
        settings=dict(zones=[dict(name='Main',capacity=1,polygon=None)])
        self.assertEqual(self.client.post(f'/api/sessions/{key}/start',json=settings,headers=self.headers).status_code,409)
        self.assertEqual(self.client.delete(f'/api/sessions/{key}',headers=self.headers).status_code,409)
        self.client.post(f'/api/sessions/{key}/cancel',headers=self.headers)
        self.assertTrue((path/'cancel').exists())
        atomic_json(path/'status.json',dict(state='complete'))
        atomic_json(path/'result.json',dict(settings={}))
        self.assertEqual(self.client.post(f'/api/sessions/{key}/start',json=settings,headers=self.headers).status_code,409)
    def test_retention_only_deletes_managed_expired_inactive_sessions(self):
        old=self.sessions/('a'*32);old.mkdir();atomic_json(old/'metadata.json',dict(created=1))
        running=self.sessions/('b'*32);running.mkdir();atomic_json(running/'metadata.json',dict(created=1));atomic_json(running/'status.json',dict(state='running'))
        unrelated=self.root/'private.mp4';unrelated.write_bytes(b'keep')
        web.cleanup()
        self.assertFalse(old.exists());self.assertTrue(running.exists());self.assertTrue(unrelated.exists())
    def test_restart_recovers_interrupted_job_and_removes_partial_upload(self):
        path=self.sessions/('a'*32);path.mkdir()
        atomic_json(path/'metadata.json',dict(created=1))
        atomic_json(path/'status.json',dict(state='running'))
        (path/'annotated.mp4').write_bytes(b'partial')
        (path/'source.mp4').write_bytes(b'original')
        orphan=self.sessions/('b'*32);orphan.mkdir();(orphan/'source.mp4').write_bytes(b'partial')
        web.recover_jobs()
        self.assertEqual(web.read(path/'status.json')['state'],'failed')
        self.assertFalse((path/'annotated.mp4').exists())
        self.assertTrue((path/'source.mp4').exists())
        self.assertFalse(orphan.exists())
        atomic_json(path/'status.json',dict(state='running'))
        atomic_json(path/'result.json',dict(processed_frames=30))
        (path/'annotated.mp4').write_bytes(b'completed')
        web.recover_jobs()
        self.assertEqual(web.read(path/'status.json')['state'],'complete')
        self.assertTrue((path/'annotated.mp4').exists())

if __name__=='__main__':unittest.main()

# 메갈리스 (Megalith)

브라우저에서 바로 플레이하는 3D 액션 어드벤처. 기억을 잃은 로빈이 스톤헨지 유적에서 깨어나
마을 엘렌하르와 그 주변 자연, 바다 위의 성으로 나아간다. three.js(r128)로 렌더링하고,
모든 3D 자산은 Blender 스크립트와 Poly Haven(CC0) 사진 스캔으로 만든다.

- 플레이: Claude 아티팩트 `https://claude.ai/artifact/RXedGzfxvd49EXVy6kSZa6` (개발자 시작 메뉴에서 1–4번 선택)
- 로컬 실행: `python tools/build_megalith.py` 후 `web/megalith/dist/index.html`을 로컬 서버로 연다
  (예: 저장소 루트에서 `python -m http.server 8765` → `http://localhost:8765/web/megalith/dist/index.html`)

## 폴더 구조

```
web/megalith/          게임 소스 (아티팩트로 배포되는 페이지)
  head.html            HTML/CSS 골격과 UI
  main.js              게임 전체: 렌더러, 조작, 카메라, 대화, 유적·마을·자연·성·몬스터·음악
  story.json           대사와 이벤트
  villagers.json       마을 주민 정의(위치, 소품, 대사)
  tex/                 마을 표면 디테일 텍스처 (tools/hq_textures.py)
  ground/              지면 스캔 재질 5종 + 풀 카드 (tools/pack_ground.py)
  sky/                 HDRI 하늘: 배경 4K, 조명용 HDR, 태양 방향 (tools/pack_sky.py)
  music/               장소별 배경음악 (유적 / 마을 / 숲과 들판)

tools/                 자산 생성·빌드 스크립트 (아래 파이프라인 참고)
assets_src/            생성된 중간 자산 (GLB, 텍스처, 데이터)
  hq/                  실사 자연·성: trees_*.glb, props_*.glb, castle.glb, land.json, land_splat.png
  monsters/            촉수 괴물(horror.glb), 얼음 골렘(golem.glb)
  npc/                 주민 피부·얼굴 텍스처, 소품
  region/ castle/ interiors/ items/   이전 단계의 Blender 모델 (일부는 아직 미사용)
live_src/              게임이 그대로 쓰는 원본: hero.glb(로빈), forest.glb·world.json(유적), npcs.glb
hero_game/             Godot 4 프로젝트 (로빈 이동·점프 테스트용, 웹판의 기능은 없음)
```

## 자산 파이프라인

Blender 5.3 스크립트는 `blender --background --factory-startup --python tools/<script>.py` 로 실행한다.

| 단계 | 스크립트 | 결과 |
|---|---|---|
| 스캔 내려받기 | `python tools/fetch_polyhaven.py textures\|models\|hdris 2k <id...>` | `ph/` (git 제외) |
| 지면 재질 | `python tools/pack_ground.py 2048` | `web/megalith/ground/` |
| 하늘 | `tools/pack_sky.py` (Blender) | `web/megalith/sky/` |
| 나무 (스캔 잎 촬영 → 가지 구조) | `tools/hq_trees2.py` (Blender) | `assets_src/hq/trees_broad.glb`, `trees_fir.glb` |
| 풀 카드 | `tools/hq_grass.py` (Blender) | `assets_src/hq/trees_tex/leaf_grass_*.png` |
| 바위·식물 스캔 정리 | `tools/hq_props.py` (Blender) | `assets_src/hq/props_hard.glb`, `props_soft.glb` |
| 지형·배치 | `python tools/build_land.py` | `assets_src/hq/land.json`, `land_splat.png` |
| 성·다리·대전 | `tools/hq_castle.py` (Blender) → `python tools/glb_shrink.py assets_src/hq/castle.glb assets_src/hq/castle.glb --max 1024` | `assets_src/hq/castle.glb`, `castle.json` |
| 몬스터 (고폴리 조각 → 텍스처 굽기) | `tools/hq_golem.py`, `tools/hq_horror.py` (Blender, 공용 `bl_pbr.py`) | `assets_src/monsters/*.glb` |
| 마을 표면 텍스처 | `python tools/hq_textures.py` | `web/megalith/tex/` |
| 주민 얼굴·피부 | `python tools/npc_faces.py`, `python tools/npc_textures.py`, `python tools/build_npcs.py` | `live_src/npcs.glb` |
| 최종 빌드 | `python tools/build_megalith.py` | `web/megalith/dist/` (아티팩트 업로드용) |

마을 지형·건물(`village.*`)은 다른 세션에서 만든 것으로, 배포된 아티팩트에서 받아 `raw/live/`에 둔다
(빌드 스크립트가 `dist/`로 복사한다).

## 배포 규칙 (Claude 아티팩트)

- 파일당 16 MB, 한 번에 64 MB까지 → 큰 GLB는 gzip+base64 `.b64.txt`로 나누어 싣는다.
- `.hdr`은 서빙되지 않아 base64 텍스트(`sky_env.hdr.txt`)로 싣고 페이지에서 복원한다.

## 작업 시 주의

- 이 PC의 PowerShell은 한국어 코드페이지(cp949)라 `Get-Content | Set-Content`로 UTF-8 파일을 고치면
  한글이 깨진다. 편집은 에디터나 Python으로 한다.
- Windows 경로 260자 제한: 깊은 경로에 쓸 때는 짧은 이름을 쓴다.

## 라이선스

- 사진 스캔 텍스처·모델·HDRI: Poly Haven, CC0
- 배경음악: 저장소 소유자 제공

"""Generate repository-native illustration fixtures, explicitly labelled as demo."""
import json
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parents[1] / "demo"
SCENES = [
    ("林间的白色汽车", "2019-06-14", ["tree", "car", "white", "countryside"], "大树与白色车辆，安静的郊外。"),
    ("树下的银色汽车", "2021-09-22", ["tree", "car", "silver", "countryside"], "银色车停在树旁，没有人物。"),
    ("林中的小路", "2018-05-03", ["tree", "countryside"], "绿树、小路与午后的光。"),
    ("树下的红衣身影", "2020-04-11", ["tree", "person", "red", "countryside"], "红衣人物站在树下。"),
    ("海边的落日", "2022-08-16", ["beach", "sunset"], "沙滩、海面与橙色夕阳。"),
    ("一起看海的人", "2023-07-09", ["beach", "person"], "一个人站在浅色的沙滩。"),
    ("雪山的清晨", "2017-12-25", ["mountain", "snow"], "层叠的雪山和清澈的天空。"),
    ("湖畔与群山", "2021-10-05", ["mountain", "lake", "tree"], "湖水、青山和岸边的树。"),
    ("窗边的小猫", "2024-02-20", ["cat"], "猫坐在明亮的窗边。"),
    ("草地上的小狗", "2020-03-28", ["dog", "countryside"], "小狗在草地上休息。"),
    ("生日那天", "2023-11-06", ["birthday"], "蜡烛、奶油蛋糕和生日的记忆。"),
    ("城市亮起灯的时候", "2016-09-30", ["city", "night", "car"], "城市夜景与街道上的汽车。"),
]


def tree(x, y, scale=1):
    return f'<g transform="translate({x} {y}) scale({scale})"><path d="M0 0L-6-120 10-120 18 0Z" fill="#7d745a"/><path d="M3-90 -35-140M7-73 35-135" stroke="#7d745a" stroke-width="8"/><ellipse cx="0" cy="-161" rx="67" ry="61" fill="#587354"/><ellipse cx="-34" cy="-139" rx="46" ry="41" fill="#6f8c65"/><ellipse cx="38" cy="-147" rx="47" ry="42" fill="#789268"/><ellipse cx="-12" cy="-183" rx="45" ry="30" fill="#819e71"/></g>'


def car(x, y, color):
    return f'<g transform="translate({x} {y})"><ellipse cx="100" cy="56" rx="124" ry="11" fill="#4d5847" opacity=".16"/><path d="M1 16 29 6 55-37 130-37 170 7 198 16 204 44H-2Z" fill="{color}"/><path d="M63-31 95-31 95 3H40ZM104-31 126-31 158 3H104Z" fill="#819ca0"/><path d="M3 19h197" stroke="#d4d7cc" stroke-width="3"/><circle cx="42" cy="44" r="18" fill="#49504a"/><circle cx="161" cy="44" r="18" fill="#49504a"/><circle cx="42" cy="44" r="9" fill="#abb1a6"/><circle cx="161" cy="44" r="9" fill="#abb1a6"/><path d="M-1 20h17v11H0m181-11h19v11h-19" fill="#eddfb2"/></g>'


def person(x, y, color="#b78a73"):
    return f'<g transform="translate({x} {y})"><circle cy="-91" r="16" fill="#c8a28a"/><path d="M-13-69Q0-80 14-69L21-13H-22Z" fill="{color}"/><path d="M-10-14-12 27M10-14 13 27" stroke="#576455" stroke-width="12" stroke-linecap="round"/></g>'


def illustration(tags):
    scene = '<rect width="800" height="600" fill="#cfdbcd"/><path d="M0 285Q190 195 410 294T800 255V600H0Z" fill="#a9bd96"/><path d="M0 370Q230 308 470 361T800 326V600H0Z" fill="#8ca57d"/>'
    if "beach" in tags:
        scene = '<rect width="800" height="600" fill="#c4d8d9"/><rect y="279" width="800" height="220" fill="#8dafb1"/><path d="M0 457Q180 412 415 447T800 431V600H0Z" fill="#e2d4b7"/><path d="M0 445Q180 400 415 435T800 419" fill="none" stroke="#e9eee5" stroke-width="12"/>'
        if "sunset" in tags:
            scene = '<rect width="800" height="600" fill="#ddbea1"/><circle cx="524" cy="249" r="53" fill="#f0d2a1"/><rect y="302" width="800" height="220" fill="#91acac"/><path d="M0 450Q200 411 432 470T800 452V600H0Z" fill="#e1ceb0"/><path d="M0 445Q200 406 432 465T800 447" stroke="#e5e7d5" stroke-width="9" fill="none"/><path d="m490 316-15 135h115l-35-135" fill="#d4cbb0" opacity=".5"/>'
        if "person" in tags:
            scene += person(449, 470, "#8c9e88")
    if "mountain" in tags:
        scene = '<rect width="800" height="600" fill="#c1d0d0"/><path d="M-80 454 203 122 424 430 586 163 860 479Z" fill="#81998d"/><path d="m203 122-90 108 62-8 36 30 35-25 43 32Z" fill="#e2e8df"/><path d="m586 163-63 103 52-14 27 23 31-25 45 22Z" fill="#e2e8df"/><path d="M0 440 305 225 511 470 710 294 900 500V600H0Z" fill="#647f72"/>'
        if "snow" in tags:
            scene += '<path d="m305 225-108 104 72-2 39 38 36-20 57 26Z" fill="#eff1e8"/><path d="M0 497Q200 470 421 515T800 480V600H0Z" fill="#e8ece2"/>'
        if "lake" in tags:
            scene += '<path d="M0 446Q200 467 432 440T800 451V600H0Z" fill="#98b4ae"/><path d="M180 525h203m118-20h173" stroke="#d0dcd1" stroke-width="4"/>' + tree(88, 476, .68)
    if "countryside" in tags:
        scene += '<path d="M394 334 473 334 651 600H137Z" fill="#c1b998"/>'
    if "tree" in tags and "lake" not in tags:
        scene += tree(194, 413, 1.36) + tree(695, 365, .68)
    if "car" in tags and "night" not in tags:
        scene += car(391, 410, "#ecece0" if "white" in tags else "#b2b9b1")
    if "person" in tags and "beach" not in tags:
        scene += person(468, 431, "#b66d57")
    if "cat" in tags:
        scene = '<rect width="800" height="600" fill="#e5dccb"/><rect x="100" y="55" width="440" height="367" rx="8" fill="#b6cdcb"/><path d="M320 55v367M100 231h440" stroke="#f5eee0" stroke-width="16"/><path d="M0 476h800v124H0" fill="#c3b397"/><path d="M447 496q-70 2-25-95 65-87 90 33 7 40-16 64Z" fill="#ab967d"/><path d="m419 376 4-62 35 29 30-4 36-28 2 73" fill="#ab967d"/><ellipse cx="475" cy="380" rx="56" ry="48" fill="#ab967d"/><path d="M513 473q100 47 76-71" stroke="#ab967d" stroke-width="27" stroke-linecap="round" fill="none"/><path d="m453 382 10 1m27-1 10-1" stroke="#546457" stroke-width="5"/><path d="m470 401 7 5 7-6" fill="#796e64"/>'
    if "dog" in tags:
        scene += '<ellipse cx="404" cy="452" rx="92" ry="39" fill="#c5ad83"/><path d="m345 460-7 52m100-53 11 50" stroke="#c5ad83" stroke-width="18" stroke-linecap="round"/><ellipse cx="478" cy="412" rx="48" ry="50" fill="#d2b98e"/><ellipse cx="448" cy="405" rx="15" ry="42" fill="#a98d65"/><ellipse cx="512" cy="409" rx="15" ry="38" fill="#a98d65"/><path d="M485 431h17" stroke="#635e4a" stroke-width="9" stroke-linecap="round"/><circle cx="484" cy="405" r="4" fill="#635e4a"/><path d="M319 444q-48-8-43-61" stroke="#c5ad83" stroke-width="17" stroke-linecap="round" fill="none"/>'
    if "birthday" in tags:
        scene = '<rect width="800" height="600" fill="#ddd5c2"/><circle cx="138" cy="118" r="45" fill="#c0c9b2"/><circle cx="656" cy="169" r="59" fill="#bdc9bb"/><path d="M0 414h800v186H0" fill="#b5a68a"/><ellipse cx="402" cy="482" rx="199" ry="32" fill="#e3ddcd"/><path d="M245 321h310v140q-155 41-310 0Z" fill="#c8997f"/><ellipse cx="400" cy="323" rx="155" ry="34" fill="#f0e7d2"/><path d="M250 339q15 50 31 0 15 40 31 4 16 49 31 0 16 39 31 1 16 50 31 0 16 40 31 0 16 50 31 0 16 40 31 0" fill="#f0e7d2"/>'
        for x in [325, 398, 474]:
            scene += f'<path d="M{x} 315v-71" stroke="#799784" stroke-width="10"/><path d="M{x} 244q-22-21 0-39 22 18 0 39" fill="#e4b36c"/>'
    if "night" in tags:
        scene = '<rect width="800" height="600" fill="#5c7474"/><circle cx="610" cy="114" r="32" fill="#d2d2b4"/><path d="M0 463h800v137H0" fill="#56635d"/>'
        for x, y, w in [(15, 200, 110), (142, 118, 124), (283, 250, 105), (406, 178, 117), (543, 244, 102), (659, 165, 120)]:
            scene += f'<rect x="{x}" y="{y}" width="{w}" height="{470-y}" fill="#78887a"/>'
            for wx in range(x+16, x+w-10, 29):
                for wy in range(y+22, 446, 42):
                    scene += f'<rect x="{wx}" y="{wy}" width="12" height="19" fill="#d0c39a" opacity=".8"/>'
        scene += '<path d="M0 555h800" stroke="#a8b29a" stroke-width="4" stroke-dasharray="45 38"/>' + car(333, 493, "#b5b7a3")
    return '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600" viewBox="0 0 800 600"><defs><filter id="grain"><feTurbulence type="fractalNoise" baseFrequency=".7" numOctaves="3" stitchTiles="stitch"/><feColorMatrix type="saturate" values="0"/><feComponentTransfer><feFuncA type="linear" slope=".04"/></feComponentTransfer><feBlend in="SourceGraphic" mode="multiply"/></filter></defs>' + scene + '<rect width="800" height="600" fill="#efe8d6" opacity=".04" filter="url(#grain)"/></svg>'


if __name__ == "__main__":
    DIRECTORY.mkdir(exist_ok=True)
    manifest = []
    for i, (title, date, tags, description) in enumerate(SCENES, 1):
        filename = f"scene-{i:02}.svg"
        (DIRECTORY / filename).write_text(illustration(tags))
        manifest.append({"id": f"demo-{i:02}", "file": filename, "title": title, "date": date,
                         "tags": tags, "description": description})
    (DIRECTORY / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("Created 12 labelled demo illustrations.")

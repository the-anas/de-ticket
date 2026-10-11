"""Resolve the curated destination list to station ids and coordinates.

Run from backend/:  uv run python scripts/build_destinations.py
Writes data/destinations.json. Edit SEED to add or change destinations.
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DATA_DIR  # noqa: E402
from app.transit.transitous import TransitousProvider  # noqa: E402

# (display name, station search text, region, tags, blurb)
SEED = [
    ("Heidelberg", "Heidelberg Hbf", "Baden-Württemberg", ["old town", "castle"], "Castle ruin above the Neckar, the Philosophenweg and a compact old town."),
    ("Freiburg im Breisgau", "Freiburg (Breisgau) Hbf", "Baden-Württemberg", ["old town", "black forest"], "Sunny university city and the usual gateway to the southern Black Forest."),
    ("Konstanz", "Konstanz", "Baden-Württemberg", ["lake", "old town"], "Lake Constance city on the Swiss border with an intact medieval centre."),
    ("Tübingen", "Tübingen Hbf", "Baden-Württemberg", ["old town", "university"], "Half-timbered riverfront, punting on the Neckar and a hilltop castle."),
    ("Stuttgart", "Stuttgart Hbf", "Baden-Württemberg", ["city", "museums"], "Mercedes and Porsche museums, vineyards inside the city limits."),
    ("Lindau", "Lindau-Insel", "Bavaria", ["lake", "island"], "Island old town on Lake Constance with a harbour view of the Alps."),
    ("Füssen", "Füssen", "Bavaria", ["castle", "alps"], "End of the line for Neuschwanstein and Hohenschwangau."),
    ("Garmisch-Partenkirchen", "Garmisch-Partenkirchen", "Bavaria", ["alps", "hiking"], "Base for the Zugspitze, Partnach gorge and Alpine hiking."),
    ("Berchtesgaden", "Berchtesgaden Hbf", "Bavaria", ["alps", "lake"], "Königssee and the national park in the far south-east corner."),
    ("München", "München Hbf", "Bavaria", ["city", "museums"], "Bavarian capital: Englischer Garten, museums and beer halls."),
    ("Regensburg", "Regensburg Hbf", "Bavaria", ["old town", "unesco"], "Roman and medieval city on the Danube, largely undamaged in the war."),
    ("Passau", "Passau Hbf", "Bavaria", ["old town", "rivers"], "Baroque town where the Danube, Inn and Ilz meet."),
    ("Nürnberg", "Nürnberg Hbf", "Bavaria", ["city", "history"], "Imperial castle, old town walls and the Nazi party rally grounds documentation centre."),
    ("Bamberg", "Bamberg", "Bavaria", ["old town", "unesco", "beer"], "UNESCO old town on seven hills, known for smoked beer."),
    ("Würzburg", "Würzburg Hbf", "Bavaria", ["baroque", "wine"], "The Residenz palace and Franconian wine on the Main."),
    ("Rothenburg ob der Tauber", "Rothenburg ob der Tauber", "Bavaria", ["old town", "medieval"], "Walled medieval town; reached by a branch line from Steinach."),
    ("Trier", "Trier Hbf", "Rhineland-Palatinate", ["roman", "unesco"], "Germany's oldest city, with the Porta Nigra and Roman baths."),
    ("Cochem", "Cochem", "Rhineland-Palatinate", ["river", "castle", "wine"], "Moselle valley town below the Reichsburg castle."),
    ("Bacharach", "Bacharach", "Rhineland-Palatinate", ["rhine", "castle", "wine"], "Half-timbered village in the Upper Middle Rhine gorge."),
    ("Koblenz", "Koblenz Hbf", "Rhineland-Palatinate", ["rhine", "fortress"], "Deutsches Eck, where the Moselle joins the Rhine, and Ehrenbreitstein fortress."),
    ("Köln", "Köln Hbf", "North Rhine-Westphalia", ["city", "cathedral"], "The cathedral is next to the station; museums and the Rhine promenade nearby."),
    ("Aachen", "Aachen Hbf", "North Rhine-Westphalia", ["cathedral", "unesco"], "Charlemagne's cathedral and a short hop to the Dutch and Belgian borders."),
    ("Münster", "Münster (Westf) Hbf", "North Rhine-Westphalia", ["old town", "cycling"], "Arcaded Prinzipalmarkt and more bicycles than people."),
    ("Marburg", "Marburg", "Hesse", ["old town", "university"], "Steep half-timbered university town below a landgrave's castle."),
    ("Bremen", "Bremen Hbf", "Bremen", ["old town", "unesco"], "Town hall and Roland on the market square, the Schnoor quarter."),
    ("Hamburg", "Hamburg Hbf", "Hamburg", ["city", "harbour"], "Harbour, Speicherstadt and Elbphilharmonie; HVV ferries are included in the ticket."),
    ("Lübeck", "Lübeck Hbf", "Schleswig-Holstein", ["old town", "unesco"], "Hanseatic brick Gothic old town on an island, known for marzipan."),
    ("Westerland (Sylt)", "Westerland(Sylt)", "Schleswig-Holstein", ["sea", "island"], "North Sea island reached by train over the Hindenburgdamm."),
    ("Schwerin", "Schwerin Hbf", "Mecklenburg-Vorpommern", ["castle", "lake"], "Fairy-tale palace on an island among seven lakes."),
    ("Warnemünde", "Warnemünde", "Mecklenburg-Vorpommern", ["sea", "beach"], "Wide Baltic beach and lighthouse, S-Bahn from Rostock."),
    ("Stralsund", "Stralsund Hbf", "Mecklenburg-Vorpommern", ["old town", "unesco", "sea"], "Hanseatic old town and the gateway to Rügen."),
    ("Binz (Rügen)", "Ostseebad Binz", "Mecklenburg-Vorpommern", ["sea", "beach", "island"], "Resort architecture and a long beach on Rügen."),
    ("Berlin", "Berlin Hbf", "Berlin", ["city", "museums"], "The capital; the whole city network is covered by the ticket."),
    ("Potsdam", "Potsdam Hbf", "Brandenburg", ["palace", "unesco"], "Sanssouci and the Prussian palaces and parks."),
    ("Dresden", "Dresden Hbf", "Saxony", ["baroque", "museums"], "Frauenkirche, Zwinger and the Elbe meadows."),
    ("Bad Schandau", "Bad Schandau", "Saxony", ["hiking", "national park"], "Base for Saxon Switzerland's sandstone rocks; S-Bahn from Dresden."),
    ("Görlitz", "Görlitz", "Saxony", ["old town", "film"], "Germany's easternmost town, a frequent film set, facing Zgorzelec across the Neisse."),
    ("Leipzig", "Leipzig Hbf", "Saxony", ["city", "music"], "Bach, the Peaceful Revolution and a large city-centre station."),
    ("Weimar", "Weimar", "Thuringia", ["culture", "unesco"], "Goethe, Schiller and the Bauhaus in one small town."),
    ("Erfurt", "Erfurt Hbf", "Thuringia", ["old town", "medieval"], "Krämerbrücke, cathedral hill and a well-preserved medieval centre."),
    ("Eisenach", "Eisenach", "Thuringia", ["castle", "unesco"], "The Wartburg, where Luther translated the New Testament."),
    ("Quedlinburg", "Quedlinburg", "Saxony-Anhalt", ["old town", "unesco", "harz"], "More than a thousand half-timbered houses at the edge of the Harz."),
    ("Wernigerode", "Wernigerode Hbf", "Saxony-Anhalt", ["harz", "castle"], "Colourful town hall and castle; the Brocken steam railway from here needs its own ticket."),
    ("Goslar", "Goslar", "Lower Saxony", ["old town", "unesco", "harz"], "Imperial palace and the Rammelsberg mine at the Harz edge."),
    ("Salzburg (AT)", "Salzburg Hbf", "Austria", ["abroad", "old town"], "Across the border but covered up to Salzburg Hbf via Freilassing."),
    ("Basel (CH)", "Basel Bad Bf", "Switzerland", ["abroad", "city"], "Covered up to Basel Badischer Bahnhof; trams and buses in Basel are not."),
    ("Kufstein (AT)", "Kufstein", "Austria", ["abroad", "alps", "fortress"], "Fortress town in Tyrol, covered via Kiefersfelden."),
]


async def main() -> None:
    provider = TransitousProvider()
    out = []
    for name, query, region, tags, blurb in SEED:
        hits = await provider.search_stations(query, limit=5)
        if not hits:
            print(f"!! no station found for {query}", file=sys.stderr)
            continue
        st = hits[0]
        print(f"{name:28} -> {st.name} [{st.country}] {st.id}")
        out.append(
            {
                "name": name,
                "station": st.name,
                "station_id": st.id,
                "lat": round(st.lat, 5),
                "lon": round(st.lon, 5),
                "region": region,
                "tags": tags,
                "blurb": blurb,
            }
        )
    await provider.aclose()
    path = DATA_DIR / "destinations.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(out)} destinations to {path}")


if __name__ == "__main__":
    asyncio.run(main())

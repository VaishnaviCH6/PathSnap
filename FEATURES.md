# M2 Candidate Generation and Feature Definitions

## IDs and data integrity

- `trip_id`: Trip folder name.
- `observation_id`: Zero-based row position in the `.track` file.
- `candidate_arc_id`: Zero-based row position in the original `.arcs` file.
- `node_id`: Zero-based row position in the original `.nodes` file.
- The M2 pipeline does not read `.route` files or use ground-truth labels.

## Candidate generation

- Coordinates are projected from EPSG:4326 to a locally selected UTM CRS.
- A Shapely spatial index identifies nearby road segments.
- Exact point-to-segment distances are calculated in metres.
- Default search radius: 50 metres.
- Maximum candidates per observation: 8.
- Both settings are configurable.
- Candidate IDs preserve the original `.arcs` row positions.

Invalid coordinates and observations without candidates are recorded in the observation-status output.

## Candidate-level features

| Feature | Definition | Unit |
|---|---|---|
| distance_to_segment_m | GPS point to candidate segment distance | metres |
| direction_difference_deg | Minimum angle to either segment orientation | degrees |
| gps_step_distance_m | Distance from previous adjacent GPS observation | metres |
| time_difference_s | Difference from previous timestamp | seconds |
| gps_speed_mps | GPS step distance divided by positive time difference | m/s |
| segment_length_m | Road segment length | metres |
| projection_fraction | Closest position along segment | 0-1 |
| gps_heading_deg | GPS movement bearing clockwise from north | degrees |
| road_bearing_deg | Arc bearing from source node to destination node | degrees |
| gps_heading_change_deg | Smallest angular change from previous GPS heading | degrees |

Observation-level movement features are repeated across candidate rows. The first observation has no previous movement information, so applicable values are missing.

The direction feature treats road segments as usable in either direction. This assumption must be confirmed with M3 before model integration.

## Output files

- `{trip_id}_candidate_features.csv`: One row per observation-candidate pair.
- `{trip_id}_observation_status.csv`: One row per GPS observation, including candidate count and candidate status.

## Known limitations

Across the 80 training trips:

- Total observations: 208,417
- Observations with candidates: 190,187
- Observations without candidates: 18,230
- Candidate coverage: 91.25%

Trip 00000054 accounts for 16,519 observations without candidates. Investigation found that their nearest road segments were between 50.20 m and 121.94 m away.

These observations are retained in the status output with `no_candidate_within_radius`.

The default radius remains 50 metres. It has not been increased globally because a larger radius may introduce irrelevant candidates.

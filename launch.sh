police_agents=0
fire_agents=3
ambulance_agents=2
police_police_offices=0
ambulance_centers=0
fire_stations=0
precompute=False


rm -rf runs
source venv/bin/activate
python launch.py -fb $fire_agents -pf $police_agents -at $ambulance_agents -po $police_police_offices -ac $ambulance_centers -fs $fire_stations
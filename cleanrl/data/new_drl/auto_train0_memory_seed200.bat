CALL activate pj_torch
PAUSE
python ppo1a_mh_4m_2g.py --seed 200 --input-ascend 80 --memory-alpha 0.01 --discard-alpha 0.5 --gamma 0.95
python ppo1a_mh_4m_2g.py --seed 200 --input-ascend 80 --memory-alpha 0.02 --discard-alpha 0.5 --gamma 0.95
python ppo1a_mh_4m_2g.py --seed 200 --input-ascend 80 --memory-alpha 0.04 --discard-alpha 0.5 --gamma 0.95
python ppo1a_mh_4m_2g.py --seed 200 --input-ascend 80 --memory-alpha 0.08 --discard-alpha 0.5 --gamma 0.95
PAUSE
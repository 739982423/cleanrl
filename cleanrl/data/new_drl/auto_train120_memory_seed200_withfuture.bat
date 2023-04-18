CALL activate pj_torch
PAUSE
python ppo1a_mh_4m_2g_wofuture.py --seed 200 --input-ascend 120 --memory-alpha 0.001 --discard-alpha 0.5 --gamma 0.95
python ppo1a_mh_4m_2g_wofuture.py --seed 201 --input-ascend 120 --memory-alpha 0.001 --discard-alpha 0.5 --gamma 0.95
python ppo1a_mh_4m_2g_wofuture.py --seed 202 --input-ascend 120 --memory-alpha 0.001 --discard-alpha 0.5 --gamma 0.95
python ppo1a_mh_4m_2g_wofuture.py --seed 203 --input-ascend 120 --memory-alpha 0.001 --discard-alpha 0.5 --gamma 0.95
python ppo1a_mh_4m_2g_wofuture.py --seed 204 --input-ascend 120 --memory-alpha 0.001 --discard-alpha 0.5 --gamma 0.95
PAUSE
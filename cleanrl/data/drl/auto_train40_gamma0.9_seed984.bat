CALL activate pj_torch
PAUSE
python ppo1a_4m_2g.py --input-ascend 40 --seed 984 --gamma 0.9 --load-alpha 0.99
PAUSE
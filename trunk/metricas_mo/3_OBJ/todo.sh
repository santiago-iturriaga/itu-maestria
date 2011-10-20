./mass_cat_obj.sh
python gen_cortes_2d.py > cortes.log
./mass_fp_obj.sh > fp.log
./mass_plot.obj.sh
python gather_data.py

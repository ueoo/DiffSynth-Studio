{
# Optional: specify checkpoint epoch for all jobs, e.g.:
#   LORA_EPOCH=49 bash inference_blenderflowers/render_wan_all.sh

echo "dahliafull forward"
bash inference_blenderflowers/render_wan_i2v_dahliafull_forward_81f.sh &
echo "dahliafull reverse"
bash inference_blenderflowers/render_wan_i2v_dahliafull_reverse_81f.sh &

echo "daisyfull forward"
bash inference_blenderflowers/render_wan_i2v_daisyfull_forward_81f.sh &
echo "daisyfull reverse"
bash inference_blenderflowers/render_wan_i2v_daisyfull_reverse_81f.sh &

echo "hibiscusfull forward"
bash inference_blenderflowers/render_wan_i2v_hibiscusfull_forward_81f.sh &
echo "hibiscusfull reverse"
bash inference_blenderflowers/render_wan_i2v_hibiscusfull_reverse_81f.sh &

echo "lilyfull forward"
bash inference_blenderflowers/render_wan_i2v_lilyfull_forward_81f.sh &
echo "lilyfull reverse"
bash inference_blenderflowers/render_wan_i2v_lilyfull_reverse_81f.sh &

echo "rosefull forward"
bash inference_blenderflowers/render_wan_i2v_rosefull_forward_81f.sh &
echo "rosefull reverse"
bash inference_blenderflowers/render_wan_i2v_rosefull_reverse_81f.sh &

wait

exit 0
}

{
# Optional: specify checkpoint epoch for all jobs, e.g.:
#   LORA_EPOCH=49 bash inference_blenderflowers/render_wan_all.sh

echo "dahliafull forward"
bash inference_blenderflowers/render_wan_i2v_dahliafull_forward.sh &
echo "dahliafull reverse"
bash inference_blenderflowers/render_wan_i2v_dahliafull_reverse.sh &

echo "daisyfull forward"
bash inference_blenderflowers/render_wan_i2v_daisyfull_forward.sh &
echo "daisyfull reverse"
bash inference_blenderflowers/render_wan_i2v_daisyfull_reverse.sh &

echo "hibiscusfull forward"
bash inference_blenderflowers/render_wan_i2v_hibiscusfull_forward.sh &
echo "hibiscusfull reverse"
bash inference_blenderflowers/render_wan_i2v_hibiscusfull_reverse.sh &

echo "lilyfull forward"
bash inference_blenderflowers/render_wan_i2v_lilyfull_forward.sh &
echo "lilyfull reverse"
bash inference_blenderflowers/render_wan_i2v_lilyfull_reverse.sh &

echo "rosefull forward"
bash inference_blenderflowers/render_wan_i2v_rosefull_forward.sh &
echo "rosefull reverse"
bash inference_blenderflowers/render_wan_i2v_rosefull_reverse.sh &

wait

exit 0
}

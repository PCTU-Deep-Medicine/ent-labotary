
cd data
curl -L -o ./kyucapsule.zip\
  https://www.kaggle.com/api/v1/datasets/download/capsuleyolo/kyucapsule

unzip -q ./kyucapsule.zip -d ./kyucapsule
rm ./kyucapsule.zip
cd ./kyucapsule

rm -rf SEE_AI_project_all_txt
rm -rf detected_samplevideo_sh
rm -rf detected_samplevideo_short
rm all_annotation.csv
mkdir images
# Move subfolders to the root directory
mv ./SEE_AI_project_all_images/SEE_AI_project_all_images/* ./images/

rm -rf SEE_AI_project_all_images

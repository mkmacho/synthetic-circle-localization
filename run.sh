docker rm -f circle
docker build -t circle .
docker run -dt \
	-v=$(pwd)/files:/files \
	--name=circle circle
docker exec -it circle bash 

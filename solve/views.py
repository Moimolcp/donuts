
from django.http import StreamingHttpResponse, JsonResponse, HttpResponseNotFound
from django.shortcuts import render
from django.utils.crypto import get_random_string
import string, os
from . import puzzle

def solve(request):

	if request.method == 'POST':

	    files = request.FILES.getlist('files')
	    
	    temp_path = 'temp/'+get_random_string(7, allowed_chars=string.ascii_uppercase + string.digits)

	    try:
	        os.makedirs(temp_path)
	    except OSError as e:
	        if e.errno == 17:
	            print("alredy exists")
	        pass
	    name = ''
	    #name_set = {}
	    for idx, f in enumerate(files):	    	
	        ext = f.name.split('.')[-1]
	        name = 'img.' + ext
	        with open(temp_path+'/'+name, 'wb+') as destination:
	            for chunk in f.chunks():
	                destination.write(chunk)
	    
	    print(temp_path)
	    b64gif = ''
	    if name != '':
	    	b64gif = puzzle.solve(temp_path,name)
	    
	    return render(request, 'solve/index.html', {'solved':b64gif})

	if request.method == 'GET':
	    return render(request, 'solve/index.html', )

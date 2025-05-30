set(GLAD_SEARCH_PATH ${DEPENDENCIES_PATH})

find_path(GLAD_INCLUDE_PATH
        NAMES glad/glad.h
        PATHS ${GLAD_SEARCH_PATH}
        PATH_SUFFIXES include)

find_library(GLAD_LIBRARIES
        NAMES glad.a glad.lib
        PATHS ${GLAD_SEARCH_PATH}
        PATH_SUFFIXES lib)

set(GLAD_FOUND "NO")
if (GLAD_INCLUDE_PATH AND GLAD_LIBRARIES)
    set(GLAD_FOUND "YES")
    message("EXTERNAL LIBRARY 'GLAD' FOUND")
else ()
    message("ERROR: EXTERNAL LIBRARY 'GLAD' NOT FOUND")
endif (GLAD_INCLUDE_PATH AND GLAD_LIBRARIES)
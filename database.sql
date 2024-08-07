create database files;
use files;

create table text_with_image_labels(
	id int not null auto_increment,
    file_name varchar(250) not null,
    file_text longtext,
    primary key(id)
    );